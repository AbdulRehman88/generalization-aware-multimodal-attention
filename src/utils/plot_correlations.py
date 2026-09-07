# import os
# import yaml
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
#
# def plot_correlations(
#     config_rel_path="configs/config.yaml",
#     top_k=20,
#     output_subdir="corr_heatmaps"
# ):
#     # resolve project root from this script’s location
#     script_dir   = os.path.dirname(__file__)
#     project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
#     cfg_path     = os.path.join(project_root, config_rel_path)
#     cfg          = yaml.safe_load(open(cfg_path))
#
#     feat_dir  = cfg['features']['output_dir']
#     fig_dir   = cfg['figures']['output_dir']
#     shap_sel  = os.path.join(os.path.dirname(feat_dir), "selected_features")
#     out_dir   = os.path.join(fig_dir, output_subdir)
#     os.makedirs(out_dir, exist_ok=True)
#
#     for mod in cfg['modalities']:
#         feat_fp = os.path.join(feat_dir, f"{mod}_features.csv")
#         sel_fp  = os.path.join(shap_sel,  f"{mod}_top{top_k}_features.txt")
#
#         if not os.path.exists(feat_fp):
#             print(f"[Skipping] no features file for '{mod}'")
#             continue
#         if not os.path.exists(sel_fp):
#             print(f"[Skipping] no selected‐features file for '{mod}'")
#             continue
#
#         # load data and selected features
#         df = pd.read_csv(feat_fp)
#         with open(sel_fp) as f:
#             feats = [l.strip() for l in f if l.strip()]
#         sub = df[feats]
#
#         # compute Pearson correlation
#         corr = sub.corr(method="pearson")
#
#         # plot
#         plt.figure(figsize=(10,8))
#         ax = sns.heatmap(
#             corr,
#             vmin=-1, vmax=1, center=0,
#             cmap="vlag",
#             square=True,
#             cbar_kws={"shrink": .75, "label": "Pearson r"},
#             xticklabels=True,
#             yticklabels=True
#         )
#         ax.set_title(f"{mod} – Pearson Correlation (Top {top_k} Features)", fontsize=14, pad=16)
#         ax.set_xlabel("Features", fontsize=12)
#         ax.set_ylabel("Features", fontsize=12)
#         plt.xticks(rotation=45, ha="right")
#         plt.yticks(rotation=0)
#         plt.tight_layout()
#
#         out_png = os.path.join(out_dir, f"{mod}_corr_heatmap.png")
#         plt.savefig(out_png, dpi=300)
#         plt.close()
#         print(f"Saved correlation heatmap for {mod} → {out_png}")
#
# if __name__ == "__main__":
#     plot_correlations()


# import os
# import yaml
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
#
#
# def plot_correlations(
#         config_rel_path="configs/config.yaml",
#         top_k=20,
#         output_subdir="corr_heatmaps"
# ):
#     # Set global font to Times New Roman, size 14, bold
#     plt.rcParams.update({
#         'font.family': 'Times New Roman',
#         'font.size': 14,
#         'font.weight': 'bold',
#         'axes.labelweight': 'bold',
#         'axes.titlesize': 16,
#         'axes.titleweight': 'bold',
#         'axes.labelsize': 16,
#         'xtick.labelsize': 14,
#         'ytick.labelsize': 14,
#     })
#
#     # resolve project root from this script’s location
#     script_dir = os.path.dirname(__file__)
#     project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
#     cfg_path = os.path.join(project_root, config_rel_path)
#     with open(cfg_path, encoding='utf-8') as f:
#         cfg = yaml.safe_load(f)
#
#     feat_dir = cfg['features']['output_dir']
#     fig_dir = cfg['figures']['output_dir']
#     shap_sel = os.path.join(os.path.dirname(feat_dir), "selected_features")
#     out_dir = os.path.join(fig_dir, output_subdir)
#     os.makedirs(out_dir, exist_ok=True)
#
#     for mod in cfg['modalities']:
#         feat_fp = os.path.join(feat_dir, f"{mod}_features.csv")
#         sel_fp = os.path.join(shap_sel, f"{mod}_top{top_k}_features.txt")
#
#         if not os.path.exists(feat_fp):
#             print(f"[Skipping] no features file for '{mod}'")
#             continue
#         if not os.path.exists(sel_fp):
#             print(f"[Skipping] no selected‐features file for '{mod}'")
#             continue
#
#         # load data and selected features
#         df = pd.read_csv(feat_fp)
#         with open(sel_fp) as f:
#             feats = [l.strip() for l in f if l.strip()]
#         sub = df[feats]
#
#         # compute Pearson correlation
#         corr = sub.corr(method="pearson")
#
#         # plot
#         plt.figure(figsize=(14, 12))
#         ax = sns.heatmap(
#             corr,
#             vmin=-1, vmax=1, center=0,
#             cmap="vlag",
#             square=True,
#             cbar_kws={"shrink": .75, "label": "Pearson r"},
#             xticklabels=True,
#             yticklabels=True
#         )
#
#         # Bold the tick labels
#         ax.set_xticklabels(ax.get_xticklabels(), fontsize=14, fontweight='bold', fontname='Times New Roman',
#                            rotation=45, ha="right")
#         ax.set_yticklabels(ax.get_yticklabels(), fontsize=14, fontweight='bold', fontname='Times New Roman', rotation=0)
#
#         # Set title and labels (larger and bold)
#         ax.set_title(f"{mod} – Pearson Correlation (Top {top_k} Features)", fontsize=16, fontweight='bold',
#                      fontname='Times New Roman', pad=18)
#         ax.set_xlabel("Features", fontsize=16, fontweight='bold', fontname='Times New Roman', labelpad=10)
#         ax.set_ylabel("Features", fontsize=16, fontweight='bold', fontname='Times New Roman', labelpad=10)
#
#         # Make colorbar label bold and font size 16
#         cbar = ax.collections[0].colorbar
#         cbar.set_label('Pearson r', fontsize=16, fontweight='bold', fontname='Times New Roman', labelpad=10)
#         cbar.ax.tick_params(labelsize=14)
#         for l in cbar.ax.yaxis.get_ticklabels():
#             l.set_fontweight('bold')
#             l.set_fontname('Times New Roman')
#
#         plt.tight_layout()
#
#         out_png = os.path.join(out_dir, f"{mod}_corr_heatmap.png")
#         plt.savefig(out_png, dpi=300)
#         plt.close()
#         print(f"Saved correlation heatmap for {mod} → {out_png}")
#
#
# if __name__ == "__main__":
#     plot_correlations()





# ===============Latest code that I wrote to generate the corr-heatmaps for JBHI paper ==============

import os
import yaml
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_correlation_heatmaps(config_path="configs/config.yaml"):
    # === Load config
    with open(config_path, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    # Paths from config
    base_dir  = cfg["features"]["output_dir"]
    shap_dir  = cfg["shap"]["shap_dir"]
    output_dir = cfg["stat_results"]["correlation_heatmap_dir"]
    os.makedirs(output_dir, exist_ok=True)
    modalities = cfg["modalities_feature_files"]
    top_k = cfg["shap"]["top_k_heatmap"]

    # Set font properties globally
    plt.rcParams.update({
        'font.family': 'Times New Roman',
        'font.size': 16,
        'font.weight': 'bold'
    })

    for modality in modalities:
        print(f"🔍 Generating correlation heatmap for: {modality}")

        data_path = os.path.join(base_dir, f"{modality}.csv")
        shap_path = os.path.join(shap_dir, f"{modality}_SHAP_Importance.csv")

        if not os.path.exists(data_path) or not os.path.exists(shap_path):
            print(f"⚠️ Missing file for {modality}, skipping.")
            continue

        # Load data
        df = pd.read_csv(data_path)
        shap_df = pd.read_csv(shap_path)

        # Top K SHAP features
        top_feats = [f for f in shap_df["Feature"].tolist()[:top_k] if f in df.columns]
        if not top_feats:
            print(f"⚠️ No valid SHAP features found in {modality}, skipping.")
            continue

        # Compute correlation matrix
        corr_matrix = df[top_feats].corr()

        # Plot (same as original, just larger font/fig)
        plt.figure(figsize=(12, 10))
        ax = sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",  # UNCHANGED
            square=True,
            cbar=True,
            linewidths=0.5,
            linecolor='gray'
        )
        ax.set_title(
            modality.replace("_features", "").replace("_", " + "),
            fontsize=20, fontweight='bold'
        )
        ax.set_xlabel("Features", fontsize=20, fontweight='bold')
        ax.set_ylabel("Features", fontsize=20, fontweight='bold')
        plt.xticks(rotation=45, ha="right", fontsize=18, fontweight='bold')
        plt.yticks(rotation=0, fontsize=18, fontweight='bold')
        plt.tight_layout()

        # Save
        save_path = os.path.join(output_dir, f"{modality}_CorrelationHeatmap.png")
        plt.savefig(save_path, dpi=300)
        plt.savefig(save_path.replace(".png", ".svg"))
        plt.savefig(save_path.replace(".png", ".pdf"))
        plt.close()

        print(f"✅ Saved: {save_path}")

    print("✅ All individual correlation heatmaps saved.")

if __name__ == "__main__":
    plot_correlation_heatmaps()
