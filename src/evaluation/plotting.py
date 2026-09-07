import os
import joblib
import yaml
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc

# Human-readable labels
CLASS_LABELS = ['Low Attention', 'Mid Attention', 'High Attention']

# Set global font and line style properties
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 12,
    'font.weight': 'bold',
    'axes.linewidth': 1.5,  # Make lines bolder (black color will be used by default)
    'lines.color': 'black'  # Ensure line color is black
})


def _load_selected_features(cfg, mod, top_k):
    feat_dir = cfg['features']['output_dir']
    sel_file = os.path.join(
        os.path.dirname(feat_dir),
        'selected_features',
        f"{mod}_top{top_k}_features.txt"
    )
    with open(sel_file) as f:
        return [line.strip() for line in f if line.strip()]


def plot_confusion(mod, clf_name, cfg, top_k=20):
    # Load features CSV
    feat_dir = cfg['features']['output_dir']
    df = pd.read_csv(os.path.join(feat_dir, f"{mod}_features.csv"))
    y = df['label']
    # Restrict to SHAP-selected features
    sel_feats = _load_selected_features(cfg, mod, top_k)
    X = df[sel_feats]

    # Load trained model
    model_path = os.path.join(cfg['models']['output_dir'], f"{mod}_{clf_name}.pkl")
    model = joblib.load(model_path)

    # Predict
    preds = model.predict(X)

    # Confusion matrix
    cm = confusion_matrix(y, preds, labels=[0, 1, 2])
    disp = ConfusionMatrixDisplay(cm, display_labels=CLASS_LABELS)
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, cmap='Blues', colorbar=False)
    ax.set_title(f"{mod} – {clf_name} Confusion Matrix", fontsize=14)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()

    save_path = os.path.join(cfg['figures']['output_dir'], f"{mod}_{clf_name}_cm.png")
    plt.savefig(save_path, dpi=300)
    print(f"Confusion matrix saved at: {save_path}")  # Confirm saving
    plt.close(fig)


def plot_roc(mod, clf_name, cfg, top_k=20):
    # Load features CSV
    feat_dir = cfg['features']['output_dir']
    df = pd.read_csv(os.path.join(feat_dir, f"{mod}_features.csv"))
    y = df['label']
    # Restrict to SHAP-selected features
    sel_feats = _load_selected_features(cfg, mod, top_k)
    X = df[sel_feats]

    # Load trained model
    model_path = os.path.join(cfg['models']['output_dir'], f"{mod}_{clf_name}.pkl")
    model = joblib.load(model_path)
    if not hasattr(model, "predict_proba"):
        return

    # Predict probabilities
    y_prob = model.predict_proba(X)
    fig, ax = plt.subplots(figsize=(6, 5))
    # One-vs-rest curves
    for i, label in enumerate([0, 1, 2]):
        fpr, tpr, _ = roc_curve((y == label).astype(int), y_prob[:, i])
        ax.plot(fpr, tpr, lw=2, label=f"{CLASS_LABELS[label]} (AUC {auc(fpr, tpr):.2f})")

    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', lw=1)
    ax.set_title(f"{mod} – {clf_name} ROC Curves", fontsize=14)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.legend(loc='lower right', fontsize=10)
    plt.tight_layout()

    save_path = os.path.join(cfg['figures']['output_dir'], f"{mod}_{clf_name}_roc.png")
    plt.savefig(save_path, dpi=300)
    print(f"ROC curve saved at: {save_path}")  # Confirm saving
    plt.close(fig)
