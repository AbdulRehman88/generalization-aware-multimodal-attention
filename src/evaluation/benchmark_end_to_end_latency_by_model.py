# File: src/evaluation/benchmark_end_to_end_latency_by_model.py

import os
import sys

# ─── Ensure project root is in sys.path so “src.” imports work ───────────────
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import glob
import time
import joblib
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Import your existing feature‐extraction functions
from src.features.extract_features import extract_ecg_features, extract_eeg_features, extract_pupil_features

# def load_config(path="configs/config.yaml"):
#     with open(path, encoding="utf-8") as f:
#         return yaml.safe_load(f)

def load_config(path="configs/config.yaml"):
    # Always resolve config path relative to this script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cfg_path = os.path.join(script_dir, "..", "..", path)
    cfg_path = os.path.abspath(cfg_path)
    with open(cfg_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def generate_synthetic_signals(modality: str, fs: int, duration_sec: int = 4):
    """
    Create one synthetic 4-second segment at fs Hz for the given MODALITY (uppercase).
    Returns a dict mapping 'ECG','EEG','PUPIL' → DataFrame of raw samples.
    """
    n = int(fs * duration_sec)
    t = np.arange(n) / fs
    signals = {}

    if "ECG" in modality:
        ecg_sig = 0.5 * np.sin(2 * np.pi * 1.0 * t) + 0.05 * np.random.randn(n)
        signals["ECG"] = pd.DataFrame({"ECG_Signal": ecg_sig})

    if "EEG" in modality:
        channels = ["AF7","Fp1","Fpz","Fp2","AF8","O1","POz","O2"]
        data = {}
        for ch in channels:
            data[ch] = np.sin(2 * np.pi * 10 * t + np.random.rand()) + 0.1 * np.random.randn(n)
        signals["EEG"] = pd.DataFrame(data)

    if "PUPIL" in modality:
        x = 320 + 5 * np.sin(2 * np.pi * 0.5 * t) + 1.0 * np.random.randn(n)
        y = 240 + 5 * np.cos(2 * np.pi * 0.5 * t) + 1.0 * np.random.randn(n)
        r = 3 + 0.2 * np.random.randn(n)
        signals["PUPIL"] = pd.DataFrame({"x": x, "y": y, "r": r})

    return signals

def extract_features_from_signals(signals: dict, fs: int) -> dict:
    """
    Given a dict with keys possibly among {"ECG","EEG","PUPIL"} → DataFrame,
    run the corresponding extract_*_features and merge into one dict of feature values.
    """
    feats = {}
    if "ECG" in signals:
        feats.update(extract_ecg_features(signals["ECG"], fs))
    if "EEG" in signals:
        feats.update(extract_eeg_features(signals["EEG"], fs))
    if "PUPIL" in signals:
        feats.update(extract_pupil_features(signals["PUPIL"], fs))
    return feats

def parse_model_filename(fname: str):
    """
    Given 'ecg_eeg_xgb.pkl' → ('ECG_EEG', 'XGB').
    """
    base = os.path.basename(fname).replace(".pkl", "")
    parts = base.split("_")
    clf = parts[-1].upper()
    modality = "_".join(parts[:-1]).upper()
    return modality, clf

def get_topk_features(modality: str, shap_dir: str, top_k: int = 20):
    """
    Locate "<modality>_top*_features.txt" under shap_dir (case‐insensitive),
    read lines, and return the first top_k feature names.
    """
    pattern = os.path.join(shap_dir, f"{modality}_top*_features.txt")
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f"No SHAP file for '{modality}' in {shap_dir}")
    txt_path = matches[0]
    with open(txt_path, encoding="utf-8") as f:
        lines = f.read().splitlines()
    return lines[:top_k]

def benchmark_end_to_end_by_model():
    cfg = load_config()
    fs = cfg["preprocessing"]["resample_rate"]
    model_dir = cfg["models"]["output_dir"]
    shap_dir = os.path.join(os.path.dirname(cfg["features"]["output_dir"]), "selected_features")

    # Allowed modalities and classifiers (uppercased)
    allowed_modalities = set(m.upper() for m in cfg["modalities"])
    allowed_clfs = {"XGB", "RF", "ET", "LGBM", "CAT"}

    results = []
    n_repeat = 50

    # 1) Gather all .pkl files
    all_models = sorted(glob.glob(os.path.join(model_dir, "*.pkl")))
    if not all_models:
        print("[ERROR] No .pkl models found under outputs/models/")
        return

    for model_path in all_models:
        modality, clf = parse_model_filename(model_path)

        # 2) Only process if modality is in config AND classifier is one of the five
        if modality not in allowed_modalities or clf not in allowed_clfs:
            continue

        # 3) Load SHAP top-20 for this modality
        try:
            top_feats = get_topk_features(modality, shap_dir, top_k=20)
        except FileNotFoundError:
            print(f"[WARN] No SHAP file for {modality}, skipping {clf}")
            continue

        # 4) Preload model once
        model = joblib.load(model_path)

        # 5) Warm-up pass
        signals_wu = generate_synthetic_signals(modality, fs)
        feat_wu = extract_features_from_signals(signals_wu, fs)
        X_wu = pd.DataFrame([feat_wu])
        X_sel_wu = X_wu[top_feats]
        scaler_wu = StandardScaler().fit(X_sel_wu)
        X_scaled_wu = scaler_wu.transform(X_sel_wu)
        _ = model.predict(X_scaled_wu)

        # 6) Timed loop
        latencies = []
        for _ in range(n_repeat):
            signals = generate_synthetic_signals(modality, fs)

            t0 = time.perf_counter()
            feat_dict = extract_features_from_signals(signals, fs)
            X_df = pd.DataFrame([feat_dict])
            X_sel = X_df[top_feats]
            scaler = StandardScaler().fit(X_sel)
            X_scaled = scaler.transform(X_sel)
            _ = model.predict(X_scaled)
            t1 = time.perf_counter()

            latencies.append((t1 - t0) * 1e3)

        mean_lat = np.mean(latencies)
        std_lat  = np.std(latencies)

        results.append({
            "modality": modality,
            "classifier": clf,
            "mean_latency_ms": mean_lat,
            "std_latency_ms": std_lat
        })
        print(f"{modality}+{clf}: {mean_lat:.2f} ± {std_lat:.2f} ms")

    # 7) Build DataFrame & save CSV
    df_res = pd.DataFrame(results)
    out_csv = os.path.join(cfg["summary"]["output_dir"], "end_to_end_latency_by_model.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df_res.to_csv(out_csv, index=False)
    print(f"Saved CSV ➔ {out_csv}")

    if not df_res.empty:
        # 8) Cap error bars at 90% of the mean
        err_bars = np.minimum(df_res["std_latency_ms"], df_res["mean_latency_ms"] * 0.9)

        plt.rcParams["font.family"] = "Times New Roman"
        plt.rcParams["font.size"] = 18
        fig, ax = plt.subplots(figsize=(14, 7))

        labels = df_res["modality"] + "+" + df_res["classifier"]
        x_pos = np.arange(len(labels))

        ax.bar(
            x_pos,
            df_res["mean_latency_ms"],
            yerr=err_bars,
            capsize=4,
            color=plt.get_cmap("Pastel1").colors
        )
        # Title and axis labels
        ax.set_title("End-to-End Latency by Modality+Classifier (ms)", fontsize=20, fontweight="bold")
        ax.set_xlabel("Modality + Classifier", fontsize=18, fontweight="bold")
        ax.set_ylabel("Latency (ms)", fontsize=18, fontweight="bold")

        # Axis ticks
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=18, fontweight="bold")
        ax.tick_params(axis='y', labelsize=18)
        ax.tick_params(axis='x', labelsize=16)

        plt.tight_layout()

        out_fig = os.path.join(cfg["figures"]["output_dir"], "end_to_end_latency_by_model.png")
        os.makedirs(os.path.dirname(out_fig), exist_ok=True)
        fig.savefig(out_fig, dpi=300)
        plt.close(fig)
        print(f"Saved bar plot ➔ {out_fig}")

if __name__ == "__main__":
    benchmark_end_to_end_by_model()
