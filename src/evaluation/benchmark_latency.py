import os
import glob
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def run_benchmark(config):
    # ─── Paths & settings ────────────────────────────────────────────────
    model_dir    = config['models']['output_dir']
    feat_dir     = config['features']['output_dir']
    shap_txt_dir = os.path.join(os.path.dirname(feat_dir), 'selected_features')
    out_csv      = config['evaluation']['benchmark']['output']['csv']
    out_plot     = config['evaluation']['benchmark']['output']['plot']
    n_load       = config['evaluation']['benchmark']['n_load']
    n_infer      = config['evaluation']['benchmark']['n_infer']
    n_total      = config['evaluation']['benchmark']['n_total']

    # …
    SEED = 42
    RESULTS = []

    # ─── Helper: pick only your five classifiers, skip any '_best' files ──
    allowed = {'xgb','rf','et','lgbm','cat'}
    all_models = sorted(glob.glob(os.path.join(model_dir, '*.pkl')))
    model_paths = []
    for p in all_models:
        base = os.path.basename(p).lower()
        if 'best' in base:
            continue
        name = base[:-4]
        parts = name.split('_')
        if parts[-1] in allowed:
            model_paths.append(p)

    # ─── Main Benchmark Loop ─────────────────────────────────────────────
    for model_path in model_paths:
        # parse modality & classifier
        stem = os.path.basename(model_path)[:-4]
        parts = stem.split('_')
        clf     = parts[-1].upper()
        modality= "_".join(parts[:-1]).upper()
        # …


        # 1) Load model
        mdl = joblib.load(model_path)

        # 2) Determine feature names
        if hasattr(mdl, 'feature_names_in_'):
            feat_names = list(mdl.feature_names_in_)
        else:
            # fallback to SHAP-selected .txt
            pattern = os.path.join(shap_txt_dir, f"{modality}_top*_features.txt")
            matches = glob.glob(pattern)
            if not matches:
                raise FileNotFoundError(f"No SHAP .txt for {modality} (looked at {pattern})")
            feat_names = open(matches[0], encoding='utf-8').read().splitlines()

        # 3) Sample one matching feature vector
        csv_file = os.path.join(feat_dir, f"{modality}_features.csv")
        if not os.path.isfile(csv_file):
            raise FileNotFoundError(f"Missing features CSV: {csv_file}")
        df = pd.read_csv(csv_file)
        df = df.drop(columns=[c for c in ('segment_id','Segment_ID','label','Label') if c in df],
                     errors='ignore')
        missing = set(feat_names) - set(df.columns)
        if missing:
            raise ValueError(f"Model {modality}+{clf} expects missing cols: {missing}")
        X0 = df[feat_names].sample(n=1, random_state=SEED).values.reshape(1, -1)

        # ─── MODEL-LOAD timing ────────────────────────────────────────────
        lt = []
        for _ in range(n_load):
            t0 = time.perf_counter()
            _ = joblib.load(model_path)
            lt.append(time.perf_counter() - t0)
        load_mean, load_std = np.mean(lt), np.std(lt)

        # ─── FEATURE-PREP timing ──────────────────────────────────────────
        scaler = StandardScaler().fit(X0)
        t0 = time.perf_counter()
        Xs = scaler.transform(X0)
        prep_ms = (time.perf_counter() - t0) * 1e3

        # ─── INFERENCE-only timing ───────────────────────────────────────
        it = []
        for _ in range(n_infer):
            t0 = time.perf_counter()
            _ = mdl.predict(Xs)
            it.append(time.perf_counter() - t0)
        infer_mean, infer_std = np.mean(it), np.std(it)

        # ─── TOTAL timing ────────────────────────────────────────────────
        tt = []
        for _ in range(n_total):
            t0 = time.perf_counter()
            m2 = joblib.load(model_path)
            X2 = StandardScaler().fit(X0).transform(X0)
            _ = m2.predict(X2)
            tt.append(time.perf_counter() - t0)
        total_mean, total_std = np.mean(tt), np.std(tt)

        RESULTS.append({
            'model':           f"{modality}+{clf}",
            'load_ms_mean':    load_mean*1e3,
            'load_ms_std':     load_std*1e3,
            'prep_ms':         prep_ms,
            'infer_ms_mean':   infer_mean*1e3,
            'infer_ms_std':    infer_std*1e3,
            'total_ms_mean':   total_mean*1e3,
            'total_ms_std':    total_std*1e3,
        })

    # ─── Save summary CSV ─────────────────────────────────────────────────
    df_res = pd.DataFrame(RESULTS)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df_res.to_csv(out_csv, index=False)
    print(f"Saved latency summary ➔ {out_csv}")

    # ─── Plot bar chart ───────────────────────────────────────────────────
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size']   = 12
    fig, ax = plt.subplots(figsize=(14,6))
    ax.bar(
        df_res['model'],
        df_res['total_ms_mean'],
        yerr=df_res['total_ms_std'],
        capsize=4,
        color=plt.get_cmap('Pastel1').colors
    )
    ax.set_title('Average Total Latency per Model (ms)', fontweight='bold')
    ax.set_xlabel('Modality + Classifier', fontweight='bold')
    ax.set_ylabel('Latency (ms)', fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_plot), exist_ok=True)
    fig.savefig(out_plot, dpi=300)
    plt.close(fig)
    print(f"Saved latency plot ➔ {out_plot}")
