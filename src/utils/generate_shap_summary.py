# File: src/utils/generate_shap_summary.py

import os
import joblib
import yaml
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

# ── CONFIG ─────────────────────────────────────────────────────────────────────
# Adjust path to your config if needed:
CONFIG_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "configs", "config.yaml")
)
with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

FEATURE_DIR = cfg['features']['output_dir']
MODEL_DIR   = cfg['models']['output_dir']
FIG_DIR     = cfg['figures']['output_dir']
MODALITIES  = cfg['modalities']
CLASS_LABELS = ['Low Attention','Mid Attention','High Attention']
CLASS_COLORS = ['#4C72B0','#55A868','#C44E52']  # blue/green/red

# ── MAIN ───────────────────────────────────────────────────────────────────────
def main(max_display=20):
    out_subdir = os.path.join(FIG_DIR, "shap_summary_NeW")
    os.makedirs(out_subdir, exist_ok=True)

    for mod in MODALITIES:
        feat_fp  = os.path.join(FEATURE_DIR, f"{mod}_features.csv")
        model_fp = os.path.join(MODEL_DIR,   f"{mod}_xgb.pkl")
        if not os.path.exists(feat_fp) or not os.path.exists(model_fp):
            print(f"[Skipping] {mod}: missing feature or model file.")
            continue

        # 1) load
        df    = pd.read_csv(feat_fp)
        X     = df.drop(columns=['segment_id','label'])
        model = joblib.load(model_fp)

        # 2) explain
        explainer = shap.Explainer(model, X)
        sv        = explainer(X)
        vals      = sv.values  # (n_samples, n_features, n_classes)

        # 3) per-class means
        # ensure 3-dimensional: (samples, feats, classes)
        if vals.ndim == 2:
            # regression/binary → pad into 3 classes
            vals = np.stack([np.zeros_like(vals),
                             np.zeros_like(vals),
                             vals], axis=2)
        mean_pc = np.abs(vals).mean(axis=0)  # shape (n_feats, 3)

        # 4) rank by total importance
        total = mean_pc.sum(axis=1)
        idxs  = np.argsort(total)[::-1][:max_display]
        feat_names = X.columns.tolist()
        top_feats  = [feat_names[i] for i in idxs]
        bar_data    = mean_pc[idxs, :]  # (top_k, 3)

        # 5) save CSV
        out_csv = os.path.join(FEATURE_DIR, f"{mod}_perclass_shap.csv")
        pd.DataFrame({
            'Feature': top_feats,
            'Low':   bar_data[:,0],
            'Mid':   bar_data[:,1],
            'High':  bar_data[:,2],
        }).to_csv(out_csv, index=False)
        print(f"→ Saved per-class SHAP CSV: {out_csv}")

        # 6) plot
        plt.figure(figsize=(8, len(top_feats)*0.3 + 1))
        y = np.arange(len(top_feats))
        left = np.zeros(len(top_feats))
        for ci, color in enumerate(CLASS_COLORS):
            plt.barh(
                y, bar_data[:,ci],
                left=left, color=color,
                edgecolor='white', height=0.6,
                label=CLASS_LABELS[ci]
            )
            left += bar_data[:,ci]

        plt.yticks(y, top_feats, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.xlabel("Mean |SHAP value|", fontweight='bold')
        plt.ylabel("Feature",        fontweight='bold')
        plt.title(f"{mod} SHAP Summary (Top {len(top_feats)})", fontweight='bold')
        plt.legend(loc='lower right', frameon=False)
        plt.tight_layout()

        out_png = os.path.join(out_subdir, f"{mod}_shap_summary.png")
        plt.savefig(out_png, dpi=300)
        plt.close()
        print(f"→ Saved stacked SHAP plot:   {out_png}")

if __name__ == "__main__":
    main()
