import os
import yaml
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap

# Human‐readable labels & research-paper palette
CLASS_LABELS = ['Low Attention', 'Mid Attention', 'High Attention']
CLASS_COLORS = ['#4C72B0',  # deep blue
                '#55A868',  # medium green
                '#C44E52']  # brick red

def plot_shap_summaries(
    config_rel_path="configs/config.yaml",
    max_display=5,
    output_subdir="shap_summary"
):
    # 1) locate project root & load config
    script_dir   = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    cfg_path     = os.path.join(project_root, config_rel_path)
    cfg          = yaml.safe_load(open(cfg_path))

    feat_dir  = cfg['features']['output_dir']
    model_dir = cfg['models']['output_dir']
    fig_dir   = cfg['figures']['output_dir']
    shap_dir  = os.path.join(fig_dir, output_subdir)
    os.makedirs(shap_dir, exist_ok=True)

    for mod in cfg['modalities']:
        try:
            feat_fp  = os.path.join(feat_dir,  f"{mod}_features.csv")
            model_fp = os.path.join(model_dir, f"{mod}_xgb.pkl")

            if not os.path.exists(feat_fp):
                print(f"[Skipping] no feature file for modality '{mod}'")
                continue
            if not os.path.exists(model_fp):
                print(f"[Skipping] no XGB model for modality '{mod}'")
                continue

            # 2) load data & model
            df    = pd.read_csv(feat_fp)
            X     = df.drop(columns=['segment_id','label'])
            model = joblib.load(model_fp)

            # 3) compute SHAP values
            explainer = shap.Explainer(model, X)
            sv        = explainer(X)
            vals      = sv.values  # either (n, f, c) or (n, f)

            # if 2D (binary/regression), pack into 3rd class only
            if vals.ndim == 2:
                vals = np.stack([
                    np.zeros_like(vals),
                    np.zeros_like(vals),
                    vals
                ], axis=2)

            if vals.ndim != 3:
                raise ValueError(f"Unexpected SHAP dims for {mod}: {vals.ndim}")

            # 4) per-feature, per-class mean-absolute
            mean_per_class = np.abs(vals).mean(axis=0)    # shape (f,3)

            # 5) rank by total importance across all 3 classes
            total_imp = mean_per_class.sum(axis=1)       # (f,)
            n_feats   = X.shape[1]
            k         = min(max_display, n_feats)
            idxs      = np.argsort(total_imp)[::-1][:k]
            idxs      = [int(i) for i in idxs]

            feat_names = X.columns.tolist()
            top_feats  = [feat_names[i] for i in idxs]
            bar_data   = mean_per_class[idxs]            # (k,3)

            # --- print out the numeric ranking ---
            print(f"\n=== {mod} SHAP summary (showing top {k} of {n_feats} features) ===")
            for rank, i in enumerate(idxs, start=1):
                print(f"{rank:2d}. {feat_names[i]:30s} → {total_imp[i]:.6f}")

            # 6) plot stacked bar
            plt.figure(figsize=(8, k*0.3 + 1))
            y_pos = np.arange(k)
            left  = np.zeros_like(y_pos, dtype=float)

            for cls_idx, color in enumerate(CLASS_COLORS):
                widths = bar_data[:, cls_idx]
                plt.barh(
                    y_pos, widths,
                    left=left,
                    color=color,
                    edgecolor='white',
                    height=0.6,
                    label=CLASS_LABELS[cls_idx]
                )
                left += widths

            plt.yticks(y_pos, top_feats)
            plt.gca().invert_yaxis()
            plt.xlabel("Mean |SHAP value|", fontsize=12)
            plt.title(f"{mod} SHAP Summary (Top {k})", fontsize=14)
            plt.legend(loc='lower right', fontsize=10)
            plt.tight_layout()

            out_png = os.path.join(shap_dir, f"{mod}_shap_summary.png")
            plt.savefig(out_png, dpi=300)
            plt.close()
            print(f"  ↳ saved plot → {out_png}")

        except Exception as e:
            print(f"[ERROR] failed on modality '{mod}': {e}")

if __name__ == "__main__":
    plot_shap_summaries()
