# File: src/evaluation/profile_ecg_latency.py

import os
import sys
import glob
import time
import joblib
import yaml
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ─── Add project root so `import src.features.extract_features` works ─────────
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.features.extract_features import extract_ecg_features

def load_config(path="configs/config.yaml"):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)

def generate_synthetic_ecg(fs, duration_sec=4):
    """
    Only ECG signal: 4 sec of 128 Hz sine + noise.
    """
    n = int(fs * duration_sec)
    t = np.arange(n) / fs
    ecg = 0.5 * np.sin(2 * np.pi * 1.0 * t) + 0.05 * np.random.randn(n)
    return pd.DataFrame({"ECG_Signal": ecg})

def profile_ecg_stage(model_path, top_feats, fs):
    """
    Run a single “synthetic‐ECG → extract_features → scale → predict” pass,
    printing the time spent in each sub‐stage.
    """
    # 1) Synthetic signal generation
    t0 = time.perf_counter()
    df_ecg = generate_synthetic_ecg(fs)
    gen_time = (time.perf_counter() - t0) * 1e3  # ms

    # 2) Feature extraction
    t1 = time.perf_counter()
    feat_dict = extract_ecg_features(df_ecg, fs)
    ext_time = (time.perf_counter() - t1) * 1e3  # ms

    # 3) Build DataFrame & pick top_feats
    X_df = pd.DataFrame([feat_dict])
    X_sel = X_df[top_feats]

    # 4) Scaling
    t2 = time.perf_counter()
    scaler = StandardScaler().fit(X_sel)
    X_scaled = scaler.transform(X_sel)
    scale_time = (time.perf_counter() - t2) * 1e3  # ms

    # 5) Model inference
    t3 = time.perf_counter()
    model = joblib.load(model_path)
    _ = model.predict(X_scaled)
    pred_time = (time.perf_counter() - t3) * 1e3  # ms

    print(f"  Generate ECG      : {gen_time:7.2f} ms")
    print(f"  Extract features  : {ext_time:7.2f} ms")
    print(f"  Scaling (z‐score) : {scale_time:7.2f} ms")
    print(f"  Load+Predict      : {pred_time:7.2f} ms")
    print(f"  → Total           : {gen_time+ext_time+scale_time+pred_time:7.2f} ms\n")

if __name__ == "__main__":
    cfg = load_config()
    fs = cfg["preprocessing"]["resample_rate"]
    model_dir = cfg["models"]["output_dir"]
    sel_dir = os.path.join(os.path.dirname(cfg["features"]["output_dir"]), "selected_features")

    # Locate the ECG XGBoost model
    model_pattern = os.path.join(model_dir, "ecg_xgb.pkl")
    matches = glob.glob(model_pattern)
    if not matches:
        raise FileNotFoundError("Cannot find ecg_xgb.pkl in outputs/models/")
    model_path = matches[0]

    # Load ECG top‐20 SHAP features
    txt_pattern = os.path.join(sel_dir, "ECG_top*_features.txt")
    txts = glob.glob(txt_pattern)
    if not txts:
        raise FileNotFoundError("Cannot find ECG_top20_features.txt in selected_features/")
    with open(txts[0], encoding="utf-8") as f:
        top_feats = f.read().splitlines()[:20]

    print("\nProfiling a single ECG pass (4 sec @ 128 Hz):")
    profile_ecg_stage(model_path, top_feats, fs)

    # To see variance, run several times:
    n_runs = 5
    for i in range(n_runs):
        print(f"Run {i+1:>2}:")
        profile_ecg_stage(model_path, top_feats, fs)
