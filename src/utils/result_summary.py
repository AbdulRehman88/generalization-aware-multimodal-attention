# import os
# import yaml
# import joblib
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.metrics import accuracy_score
#
# # 1) Resolve project root & load config
# script_dir   = os.path.dirname(__file__)
# project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
# cfg_path     = os.path.join(project_root, "configs", "config.yaml")
# cfg           = yaml.safe_load(open(cfg_path))
#
# feat_dir      = cfg['features']['output_dir']
# model_dir     = cfg['models']['output_dir']
# modalities    = cfg['modalities']
# classifiers   = ['xgb', 'rf', 'et', 'lgbm', 'cat']
# top_k         = 20
#
# # 2) Compute accuracy for each modality & classifier
# records = []
# for mod in modalities:
#     feat_fp = os.path.join(feat_dir, f"{mod}_features.csv")
#     if not os.path.exists(feat_fp):
#         continue
#
#     df        = pd.read_csv(feat_fp)
#     y_true    = df['label'].values
#
#     # load selected features
#     sel_fp    = os.path.join(
#         os.path.dirname(feat_dir),
#         'selected_features',
#         f"{mod}_top{top_k}_features.txt"
#     )
#     if not os.path.exists(sel_fp):
#         continue
#     with open(sel_fp) as f:
#         selected_feats = [l.strip() for l in f if l.strip()]
#
#     X = df[selected_feats].values
#
#     for clf in classifiers:
#         model_fp = os.path.join(model_dir, f"{mod}_{clf}.pkl")
#         if not os.path.exists(model_fp):
#             continue
#         model = joblib.load(model_fp)
#         y_pred = model.predict(X)
#         acc    = accuracy_score(y_true, y_pred)
#         records.append({
#             'modality':   mod,
#             'classifier': clf.upper(),
#             'accuracy':   acc
#         })
#
# # 3) Build DataFrame & pivot
# results_df = pd.DataFrame(records)
# pivot_df   = results_df.pivot(
#     index='modality',
#     columns='classifier',
#     values='accuracy'
# ).reindex(modalities)
#
# # 4) Plot grouped bar chart
# plt.figure(figsize=(12, 6))
# n_mod    = len(modalities)
# n_clf    = len(classifiers)
# x        = np.arange(n_mod)
# total_w  = 0.8
# bar_w    = total_w / n_clf
# offsets  = np.linspace(-total_w/2 + bar_w/2, total_w/2 - bar_w/2, n_clf)
#
# # light‐shade palette
# cmap = plt.get_cmap('Pastel1')
#
# for i, clf in enumerate([c.upper() for c in classifiers]):
#     vals = pivot_df[clf].values
#     plt.bar(x + offsets[i], vals, bar_w,
#             label=clf, color=cmap(i), edgecolor='k', linewidth=0.5)
#
# plt.xticks(x, modalities, rotation=45, ha='right')
# plt.ylim(0, 1)
# plt.ylabel("Accuracy", fontsize=14)
# plt.title("Accuracy Comparison\n(single, dual, and tri‐modal)", fontsize=16)
# plt.legend(title="Classifier", loc='lower right')
# plt.tight_layout()
#
# # 5) Save figure
# out_dir  = cfg['figures']['output_dir']
# os.makedirs(out_dir, exist_ok=True)
# out_path = os.path.join(out_dir, "accuracy_comparison_grouped.png")
# plt.savefig(out_path, dpi=300)
# plt.show()


import os
import yaml
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# 1) Resolve project root & load config
script_dir   = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
cfg_path     = os.path.join(project_root, "configs", "config.legacy.yaml")
cfg = yaml.safe_load(open(cfg_path, encoding='utf-8'))

feat_dir      = os.path.abspath(os.path.join(project_root, cfg['features']['output_dir']))
model_dir     = os.path.abspath(os.path.join(project_root, cfg['models']['output_dir']))
modalities    = cfg['modalities']
classifiers   = ['xgb', 'rf', 'et', 'lgbm', 'cat']
top_k         = 20

# 2) Compute accuracy for each modality & classifier
records = []
for mod in modalities:
    feat_fp = os.path.join(feat_dir, f"{mod}_features.csv")
    if not os.path.exists(feat_fp):
        continue

    df        = pd.read_csv(feat_fp)
    y_true    = df['label'].values

    # load selected features
    sel_fp    = os.path.join(
        os.path.dirname(feat_dir),
        'selected_features',
        f"{mod}_top{top_k}_features.txt"
    )
    if not os.path.exists(sel_fp):
        continue
    with open(sel_fp) as f:
        selected_feats = [l.strip() for l in f if l.strip()]

    X = df[selected_feats].values

    for clf in classifiers:
        model_fp = os.path.join(model_dir, f"{mod}_{clf}.pkl")
        if not os.path.exists(model_fp):
            continue
        model = joblib.load(model_fp)
        y_pred = model.predict(X)
        acc    = accuracy_score(y_true, y_pred)
        records.append({
            'modality':   mod,
            'classifier': clf.upper(),
            'accuracy':   acc
        })

# 3) Build DataFrame & pivot
results_df = pd.DataFrame(records)
pivot_df   = results_df.pivot(
    index='modality',
    columns='classifier',
    values='accuracy'
).reindex(modalities)

# Save summary CSV as requested
summary_dir = os.path.abspath(os.path.join(project_root, cfg['summary']['output_dir']))
os.makedirs(summary_dir, exist_ok=True)
results_df.to_csv(os.path.join(summary_dir, 'Accuracy_summary.csv'), index=False)

# 4) Plot grouped bar chart (with new style)
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 14,             # default font size (all text)
    'axes.labelsize': 16,        # axis labels
    'axes.labelweight': 'bold',  # axis label bold
    'axes.titlesize': 18,        # title size
    'axes.titleweight': 'bold',  # title bold
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'legend.title_fontsize': 16, # legend title font
})

plt.figure(figsize=(10, 6))
n_mod    = len(modalities)
n_clf    = len(classifiers)
x        = np.arange(n_mod)
total_w  = 0.8
bar_w    = total_w / n_clf
offsets  = np.linspace(-total_w/2 + bar_w/2, total_w/2 - bar_w/2, n_clf)

# light-shade palette
cmap = plt.get_cmap('Pastel1')

for i, clf in enumerate([c.upper() for c in classifiers]):
    vals = pivot_df[clf].values
    plt.bar(x + offsets[i], vals, bar_w,
            label=clf, color=cmap(i), edgecolor='k', linewidth=0.5)

plt.xticks(x, modalities, rotation=45, ha='right', fontweight='bold')
plt.yticks(fontsize=14, fontweight='bold')
plt.ylim(0.9, 1.0)
plt.ylabel("Accuracy", fontsize=16, fontweight='bold')
plt.xlabel("Modality", fontsize=16, fontweight='bold')
plt.title("Accuracy Comparison (single, dual, and tri-modal)", fontsize=18, fontweight='bold')
plt.legend(title="Classifier", loc='lower right')
plt.tight_layout()

# 5) Save figure
out_dir  = os.path.abspath(os.path.join(project_root, cfg['figures']['output_dir']))
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "accuracy_comparison_grouped.png")
plt.savefig(out_path, dpi=300)
plt.show()
