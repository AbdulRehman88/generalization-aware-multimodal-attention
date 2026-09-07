import os
import yaml
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import kruskal

def run_violin_and_stats(config_path="configs/config.yaml"):
    # === Load config
    with open(config_path, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    # ENFORCE Times New Roman globally for all plots
    plt.rcParams.update({
        'font.family': 'Times New Roman',
        'font.serif':  'Times New Roman',
        'font.weight': 'bold',
        'axes.labelweight': 'bold'
    })
    mpl.rcParams['mathtext.fontset'] = 'stix'

    # Modular paths
    feat_dir   = cfg["features"]["output_dir"]
    shap_dir   = cfg["shap"]["shap_dir"]
    output_dir = cfg["stat_results"]["violin_stats_dir"]
    os.makedirs(output_dir, exist_ok=True)
    modalities = cfg["modalities_feature_files"]
    top_k = cfg["shap"]["top_k_heatmap"]

    label_map = {0: "Low", 1: "Mid", 2: "High"}

    for modality in modalities:
        print(f"\n📊 Processing: {modality}")

        feat_fp = os.path.join(feat_dir, f"{modality}.csv")
        shap_fp = os.path.join(shap_dir, f"{modality}_SHAP_Importance.csv")
        if not (os.path.exists(feat_fp) and os.path.exists(shap_fp)):
            print(f"⚠️ Missing data or SHAP for {modality}, skipping.")
            continue

        df = pd.read_csv(feat_fp)
        shap_df = pd.read_csv(shap_fp)
        top_feats = [f for f in shap_df["Feature"].tolist()[:top_k] if f in df.columns]
        if not top_feats:
            print(f"⚠️ No SHAP features matched in {modality}, skipping.")
            continue

        df["Label_Name"] = df["label"].map(label_map)

        stats = {
            feat: {
                lbl: f"{df[df['Label_Name'] == lbl][feat].mean():.3f} ± {df[df['Label_Name'] == lbl][feat].std():.3f}"
                for lbl in ["Low", "Mid", "High"]
            }
            for feat in top_feats
        }
        desc_df = pd.DataFrame(stats).T
        desc_df.to_csv(os.path.join(output_dir, f"{modality}_descriptive_stats.csv"))

        kw_data = []
        for feat in top_feats:
            groups = [df[df["Label_Name"] == lbl][feat] for lbl in ["Low", "Mid", "High"]]
            stat, pval = kruskal(*groups)
            kw_data.append((feat, stat, pval))
        kw_df = pd.DataFrame(kw_data, columns=["Feature", "H-statistic", "p-value"])
        kw_df.to_csv(os.path.join(output_dir, f"{modality}_kruskal_test.csv"), index=False)

        fig, axes = plt.subplots(2, 5, figsize=(24, 10))
        axes = axes.flatten()

        for i, feat in enumerate(top_feats):
            ax = axes[i]
            sns.violinplot(x="Label_Name", y=feat, data=df, ax=ax, palette="Set2")
            ax.set_title(f"{feat}", fontsize=20, fontweight='bold')
            if i // 5 == 1:
                ax.set_xlabel("Attention Level", fontsize=20, fontweight='bold')
            else:
                ax.set_xlabel("")
            if i % 5 == 0:
                ax.set_ylabel("Feature Value", fontsize=20, fontweight='bold')
            else:
                ax.set_ylabel("")
            ax.tick_params(axis='both', which='major', labelsize=18)
            for tick in ax.get_xticklabels():
                tick.set_fontsize(18)
                tick.set_fontweight('bold')
                tick.set_fontname('Times New Roman')
            for tick in ax.get_yticklabels():
                tick.set_fontsize(18)
                tick.set_fontweight('bold')
                tick.set_fontname('Times New Roman')

        for j in range(len(top_feats), 10):
            fig.delaxes(axes[j])

        fig.suptitle(f"Top {top_k} SHAP Features — {modality}", fontsize=28, fontweight='bold')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        grid_path = os.path.join(output_dir, f"{modality}_violin_grid.png")
        plt.savefig(grid_path, dpi=300)
        plt.close()
        print(f"✅ Saved: {grid_path}")

if __name__ == "__main__":
    run_violin_and_stats()
