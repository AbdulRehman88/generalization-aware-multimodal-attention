# File: src/evaluation/benchmark_end_to_end_latency.py

import os
import sys

# ─── Add project root to sys.path so "import src.XXX" works ─────────────────
# Assumes this script lives in: <project_root>/src/evaluation/
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

# Now we can import from src.features
from src.features.extract_features import extract_ecg_features, extract_eeg_features, extract_pupil_features

def load_config(path="configs/config.yaml"):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)

def generate_synthetic_signals(modality, fs, duration_sec=4):
    """
    Generate synthetic 4-second raw data for each modality:
     - 'ECG': single‐column 'ECG_Signal'
     - 'EEG': DataFrame with 8 channels (AF7, Fp1, Fpz, Fp2, AF8, O1, POz, O2)
     - 'Pupil': DataFrame with ['x','y','r']
    """
    n_samples = int(fs * duration_sec)
    t = np.arange(n_samples) / fs
    signals = {}

    if "ECG" in modality:
        ecg = 0.5 * np.sin(2 * np.pi * 1.0 * t) + 0.05 * np.random.randn(n_samples)
        signals["ECG"] = pd.DataFrame({"ECG_Signal": ecg})

    if "EEG" in modality:
        channels = ["AF7", "Fp1", "Fpz", "Fp2", "AF8", "O1", "POz", "O2"]
        data = {}
        for ch in channels:
            data[ch] = np.sin(2 * np.pi * 10 * t + np.random.rand()) + 0.1 * np.random.randn(n_samples)
        signals["EEG"] = pd.DataFrame(data)

    if "Pupil" in modality:
        x = 320 + 5 * np.sin(2 * np.pi * 0.5 * t) + 1.0 * np.random.randn(n_samples)
        y = 240 + 5 * np.cos(2 * np.pi * 0.5 * t) + 1.0 * np.random.randn(n_samples)
        r = 3 + 0.2 * np.random.randn(n_samples)
        signals["Pupil"] = pd.DataFrame({"x": x, "y": y, "r": r})

    return signals

def extract_features_from_signals(signals, fs):
    """
    Given dictionary 'signals' with keys among 'ECG','EEG','Pupil',
    call the appropriate extract_*_features function on each and merge.
    Returns a dict of all features.
    """
    feats = {}
    if "ECG" in signals:
        f_ecg = extract_ecg_features(signals["ECG"], fs)
        feats.update(f_ecg)
    if "EEG" in signals:
        f_eeg = extract_eeg_features(signals["EEG"], fs)
        feats.update(f_eeg)
    if "Pupil" in signals:
        f_pupil = extract_pupil_features(signals["Pupil"], fs)
        feats.update(f_pupil)
    return feats

def benchmark_end_to_end():
    cfg = load_config()
    fs = cfg["preprocessing"]["resample_rate"]
    modalities = cfg["modalities"]
    model_dir = cfg["models"]["output_dir"]
    sel_dir = os.path.join(os.path.dirname(cfg["features"]["output_dir"]), "selected_features")

    results = []
    repeats = 10

    for mod in modalities:
        # 1) Find and preload the XGBoost model for this modality
        pattern = os.path.join(model_dir, f"{mod.lower()}_xgb.pkl")
        models = glob.glob(pattern)
        if not models:
            print(f"[WARN] No XGB model found for {mod}, skipping")
            continue
        model_path = models[0]
        model = joblib.load(model_path)  # preload once

        # 2) Load SHAP top-20 feature list for this modality
        txt_pattern = os.path.join(sel_dir, f"{mod}_top*_features.txt")
        txts = glob.glob(txt_pattern)
        if not txts:
            print(f"[WARN] No SHAP file for {mod}, skipping")
            continue
        with open(txts[0], encoding="utf-8") as f:
            top_feats = f.read().splitlines()[:20]

        latencies = []
        for _ in range(repeats):
            # 3) Generate synthetic raw signals
            signals = generate_synthetic_signals(mod, fs)

            start = time.perf_counter()

            # 4) Extract features
            feat_dict = extract_features_from_signals(signals, fs)

            # 5) Build DataFrame so columns align
            X_df = pd.DataFrame([feat_dict])

            # 6) Select SHAP top-20 columns
            X_sel = X_df[top_feats]

            # 7) Scale
            scaler = StandardScaler().fit(X_sel)
            X_scaled = scaler.transform(X_sel)

            # 8) Inference only (model already loaded)
            _ = model.predict(X_scaled)

            end = time.perf_counter()
            latencies.append((end - start) * 1e3)  # ms

        mean_lat = np.mean(latencies)
        std_lat = np.std(latencies)
        results.append({
            "modality": mod,
            "mean_latency_ms": mean_lat,
            "std_latency_ms": std_lat
        })
        print(f"{mod}: {mean_lat:.2f} ± {std_lat:.2f} ms")

    # Save CSV
    df_res = pd.DataFrame(results)
    out_csv = os.path.join(cfg["summary"]["output_dir"], "end_to_end_latency.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df_res.to_csv(out_csv, index=False)

    # Bar plot
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 12
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(
        df_res["modality"],
        df_res["mean_latency_ms"],
        yerr=df_res["std_latency_ms"],
        capsize=5,
        color=plt.get_cmap("Pastel1").colors
    )
    ax.set_title("End-to-End Latency by Modality (ms)", fontweight="bold")
    ax.set_xlabel("Modality", fontweight="bold")
    ax.set_ylabel("Latency (ms)", fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    out_fig = os.path.join(cfg["figures"]["output_dir"], "end_to_end_latency_comparison.png")
    os.makedirs(os.path.dirname(out_fig), exist_ok=True)
    fig.savefig(out_fig, dpi=300)
    plt.close(fig)

    print(f"Saved CSV ➔ {out_csv}")
    print(f"Saved bar plot ➔ {out_fig}")

if __name__ == "__main__":
    benchmark_end_to_end()
