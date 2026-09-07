# import os
# import glob
# import time
# import joblib
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import StandardScaler
#
# def run_benchmark(config):
#     """
#     Benchmark on-demand model-load, feature-prep (using either model.feature_names_in_
#     or SHAP top-K), inference, and total latency.
#     """
#     # ─── Paths & settings ────────────────────────────────────────────────
#     model_dir = config['models']['output_dir']
#     shap_dir  = config['evaluation']['benchmark']['shap_dir']['path']
#     feat_dir  = config['features']['output_dir']
#     out_csv   = config['evaluation']['benchmark']['output']['csv']
#     out_plot  = config['evaluation']['benchmark']['output']['plot']
#     n_load    = config['evaluation']['benchmark']['n_load']
#     n_infer   = config['evaluation']['benchmark']['n_infer']
#     n_total   = config['evaluation']['benchmark']['n_total']
#
#     SEED = 42
#     TOP_K = 20
#     RESULTS = []
#
#     def parse_info(path):
#         """Extract modality key and classifier from filename."""
#         fname = os.path.basename(path).replace('.pkl','').lower()
#         parts = fname.split('_')
#         clf = parts[-1]
#         modality = '_'.join(parts[:-1]).upper()
#         return modality, clf.upper()
#
#     # ─── Loop over every model ────────────────────────────────────────────
#     for model_path in sorted(glob.glob(os.path.join(model_dir, '*.pkl'))):
#         modality, clf = parse_info(model_path)
#
#         # Load model once
#         mdl = joblib.load(model_path)
#
#         # Determine feature names
#         if hasattr(mdl, 'feature_names_in_'):
#             feat_names = list(mdl.feature_names_in_)
#         else:
#             # fallback: top-K from SHAP importance
#             imp_glob = os.path.join(shap_dir, f"{modality}_features_SHAP_Importance.csv")
#             files = glob.glob(imp_glob)
#             if not files:
#                 raise FileNotFoundError(f"No SHAP file for {modality}, expected {imp_glob}")
#             imp_df = pd.read_csv(files[0])
#             feat_names = imp_df['Feature'].tolist()[:TOP_K]
#
#         # Sample one row matching exactly feat_names
#         full_csv = os.path.join(feat_dir, f"{modality}_features.csv")
#         full_df  = pd.read_csv(full_csv)
#         full_df  = full_df.drop(columns=[c for c in ('segment_id','Segment_ID','label','Label')
#                                           if c in full_df], errors='ignore')
#         missing = set(feat_names) - set(full_df.columns)
#         if missing:
#             raise ValueError(f"Missing features for {modality}: {missing}")
#         X0 = full_df[feat_names].sample(n=1, random_state=SEED).values.reshape(1, -1)
#
#         # 1) Model‐load timing
#         lt = []
#         for _ in range(n_load):
#             t0 = time.perf_counter()
#             _ = joblib.load(model_path)
#             lt.append(time.perf_counter() - t0)
#         load_mean, load_std = np.mean(lt), np.std(lt)
#
#         # 2) Prep timing (z‐score)
#         scaler = StandardScaler().fit(X0)
#         t0 = time.perf_counter()
#         Xs = scaler.transform(X0)
#         prep_ms = (time.perf_counter() - t0) * 1e3
#
#         # 3) Inference‐only timing
#         it = []
#         for _ in range(n_infer):
#             t0 = time.perf_counter()
#             _ = mdl.predict(Xs)
#             it.append(time.perf_counter() - t0)
#         infer_mean, infer_std = np.mean(it), np.std(it)
#
#         # 4) Total timing (load + prep + infer)
#         tt = []
#         for _ in range(n_total):
#             t0 = time.perf_counter()
#             m2 = joblib.load(model_path)
#             X2 = StandardScaler().fit(X0).transform(X0)
#             _ = m2.predict(X2)
#             tt.append(time.perf_counter() - t0)
#         total_mean, total_std = np.mean(tt), np.std(tt)
#
#         RESULTS.append({
#             'model':           f"{modality}+{clf}",
#             'load_ms_mean':    load_mean*1e3,
#             'load_ms_std':     load_std*1e3,
#             'prep_ms':         prep_ms,
#             'infer_ms_mean':   infer_mean*1e3,
#             'infer_ms_std':    infer_std*1e3,
#             'total_ms_mean':   total_mean*1e3,
#             'total_ms_std':    total_std*1e3,
#         })
#
#     # ─── Save Results ─────────────────────────────────────────────────────
#     df_res = pd.DataFrame(RESULTS)
#     os.makedirs(os.path.dirname(out_csv), exist_ok=True)
#     df_res.to_csv(out_csv, index=False)
#     print(f"Saved latency summary ➔ {out_csv}")
#
#     # ─── Plot ─────────────────────────────────────────────────────────────
#     plt.rcParams['font.family'] = 'Times New Roman'
#     plt.rcParams['font.size']   = 12
#     fig, ax = plt.subplots(figsize=(14,6))
#     ax.bar(
#         df_res['model'],
#         df_res['total_ms_mean'],
#         yerr=df_res['total_ms_std'],
#         capsize=4,
#         color=plt.get_cmap('Pastel1').colors
#     )
#     ax.set_title('Average Total Latency per Model (ms)', fontweight='bold')
#     ax.set_xlabel('Modality + Classifier', fontweight='bold')
#     ax.set_ylabel('Latency (ms)', fontweight='bold')
#     plt.xticks(rotation=45, ha='right')
#     plt.tight_layout()
#
#     os.makedirs(os.path.dirname(out_plot), exist_ok=True)
#     fig.savefig(out_plot, dpi=300)
#     plt.close(fig)
#     print(f"Saved latency plot ➔ {out_plot}")



import os
import glob
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import yaml

def run_benchmark(config):
    print("[DEBUG] run_benchmark() called")

    # ─── Paths & settings ────────────────────────────────────────────────
    model_dir = config['models']['output_dir']
    shap_dir = config['evaluation']['benchmark']['shap_dir']['path']
    feat_dir = config['features']['output_dir']
    out_csv = config['evaluation']['benchmark']['output']['csv']
    out_plot = config['evaluation']['benchmark']['output']['plot']
    n_load = config['evaluation']['benchmark']['n_load']
    n_infer = config['evaluation']['benchmark']['n_infer']
    n_total = config['evaluation']['benchmark']['n_total']
    preload = config['evaluation']['benchmark'].get('preload_models', False)

    print(f"[DEBUG] model_dir: {model_dir}")
    print(f"[DEBUG] shap_dir: {shap_dir}")
    print(f"[DEBUG] feat_dir: {feat_dir}")
    print(f"[DEBUG] out_csv: {out_csv}")
    print(f"[DEBUG] out_plot: {out_plot}")

    SEED = 42
    TOP_K = 20
    RESULTS = []

    # ─── Optional: preload all models ────────────────────────────────────
    model_cache = {}
    if preload:
        for model_path in glob.glob(os.path.join(model_dir, '*.pkl')):
            key = os.path.basename(model_path).replace('.pkl', '').lower()
            model_cache[key] = joblib.load(model_path)

    def parse_info(path):
        fname = os.path.basename(path).replace('.pkl', '').lower()
        parts = fname.split('_')
        clf = parts[-1]
        modality = '_'.join(parts[:-1]).upper()
        return modality, clf.upper(), fname

    def get_features(modality):
        feat_file = os.path.join(feat_dir, f"{modality}_features.csv")
        df = pd.read_csv(feat_file)
        df = df.drop(columns=[c for c in ('segment_id', 'Segment_ID', 'label', 'Label') if c in df], errors='ignore')
        return df

    def get_shap_features(modality):
        shap_file = os.path.join(shap_dir, f"{modality}_features_SHAP_Importance.csv")
        if not os.path.exists(shap_file):
            raise FileNotFoundError(f"Missing SHAP file for {modality}: {shap_file}")
        shap_df = pd.read_csv(shap_file)
        return shap_df['Feature'].tolist()

    # ─── Loop over every model ────────────────────────────────────────────
    for model_path in sorted(glob.glob(os.path.join(model_dir, '*.pkl'))):
        modality, clf, model_key = parse_info(model_path)
        print(f"[DEBUG] Processing model: {modality}+{clf}")

        mdl = model_cache.get(model_key) if preload else joblib.load(model_path)

        try:
            feat_names = list(mdl.feature_names_in_)
        except AttributeError:
            shap_feats = get_shap_features(modality)
            feat_names = shap_feats if len(shap_feats) <= TOP_K else shap_feats[:TOP_K]

        full_df = get_features(modality)
        missing = set(feat_names) - set(full_df.columns)
        if missing:
            raise ValueError(f"Missing features for {modality}+{clf}: {missing}")

        X0 = full_df[feat_names].sample(n=1, random_state=SEED).values.reshape(1, -1)

        lt = []
        for _ in range(n_load):
            t0 = time.perf_counter()
            _ = joblib.load(model_path)
            lt.append(time.perf_counter() - t0)
        load_mean, load_std = np.mean(lt), np.std(lt)

        scaler = StandardScaler().fit(X0)
        t0 = time.perf_counter()
        Xs = scaler.transform(X0)
        prep_ms = (time.perf_counter() - t0) * 1e3

        it = []
        for _ in range(n_infer):
            t0 = time.perf_counter()
            _ = mdl.predict(Xs)
            it.append(time.perf_counter() - t0)
        infer_mean, infer_std = np.mean(it), np.std(it)

        tt = []
        for _ in range(n_total):
            t0 = time.perf_counter()
            m2 = joblib.load(model_path)
            X2 = StandardScaler().fit(X0).transform(X0)
            _ = m2.predict(X2)
            tt.append(time.perf_counter() - t0)
        total_mean, total_std = np.mean(tt), np.std(tt)

        RESULTS.append({
            'model': f"{modality}+{clf}",
            'load_ms_mean': load_mean * 1e3,
            'load_ms_std': load_std * 1e3,
            'prep_ms': prep_ms,
            'infer_ms_mean': infer_mean * 1e3,
            'infer_ms_std': infer_std * 1e3,
            'total_ms_mean': total_mean * 1e3,
            'total_ms_std': total_std * 1e3,
        })

    print(f"[DEBUG] Number of models processed: {len(RESULTS)}")

    df_res = pd.DataFrame(RESULTS)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df_res.to_csv(out_csv, index=False)
    print(f"Saved latency summary ➔ {out_csv}")

    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 12
    fig, ax = plt.subplots(figsize=(14, 6))
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

if __name__ == "__main__":
    print("[DEBUG] LOSO Benchmark Script Started")
    config_path = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")), "configs", "config.legacy.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    run_benchmark(config)


