# File: src/shap_selection/beeswarm.py

import os
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt

def shap_beeswarm(config, modality, classifier):
    """
    Generate one beeswarm plot per class for a multiclass model,
    saving them under config['figures']['output_dir'].
    """
    # ensure output directory exists
    fig_dir = config['figures']['output_dir']
    os.makedirs(fig_dir, exist_ok=True)

    # load trained model
    model_fp = os.path.join(
        config['models']['output_dir'],
        f"{modality}_{classifier}.pkl"
    )
    model = joblib.load(model_fp)

    # load features
    feat_fp = os.path.join(
        config['features']['output_dir'],
        f"{modality}_features.csv"
    )
    df = pd.read_csv(feat_fp)
    X  = df.drop(columns=['segment_id','label'])

    # explain with SHAP
    explainer = shap.Explainer(model, X)
    sv        = explainer(X)            # sv.values shape = (n_samples, n_features, n_classes)
    base_vals = sv.base_values          # length = n_classes
    vals      = sv.values               # 3D array

    # one plot per class
    for class_idx in range(vals.shape[2]):
        # build a 2D Explanation for class_idx
        expl = shap.Explanation(
            values       = vals[:,:,class_idx],
            base_values  = base_vals[class_idx],
            data         = X.values,
            feature_names= X.columns.tolist()
        )
        plt.figure(figsize=(8,6))
        shap.plots.beeswarm(expl, show=False)
        plt.title(f"{modality} {classifier} SHAP Beeswarm (class={class_idx})", fontsize=14)

        out_png = os.path.join(
            fig_dir,
            f"{modality}_{classifier}_shap_class{class_idx}.png"
        )
        plt.tight_layout()
        plt.savefig(out_png, dpi=300)
        plt.close()
        print(f"Saved SHAP beeswarm: {out_png}")
