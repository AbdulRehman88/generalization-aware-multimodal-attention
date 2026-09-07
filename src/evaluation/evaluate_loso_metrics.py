"""
evaluate_loso_metrics.py

Perform LOSO (Leave-One-Subject-Out) cross-validation for each modality and classifier,
robust to folds that may contain more than two classes, and passing DataFrames to
tree-based models to preserve feature names (no sklearn warnings).
"""

import os
import glob
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve
)
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import yaml

# ─── Utility Functions ─────────────────────────────────────────────────────

def load_config(path="configs/config.yaml"):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)

def extract_subject_id(segment_id: str) -> str:
    """
    From "1.P01_seg0.csv" → "P01"
    """
    part = segment_id.split(".")[1]
    return part.split("_")[0]

def get_topk_features(modality: str, sel_dir: str, top_k: int = 20):
    pattern = os.path.join(sel_dir, f"{modality}_top*_features.txt")
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f"No SHAP‐selected file for '{modality}' in {sel_dir}")
    txt_path = matches[0]
    with open(txt_path, encoding="utf-8") as f:
        lines = f.read().splitlines()
    return lines[:top_k]

def plot_confusion_matrix(cm: np.ndarray, classes: list, title: str, save_path: str):
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 12
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(classes)),
        yticks=np.arange(len(classes)),
        xticklabels=classes,
        yticklabels=classes,
        ylabel="True label",
        xlabel="Predicted label",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontweight="bold")
    plt.setp(ax.get_yticklabels(), fontweight="bold")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontweight="bold"
            )
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)

def plot_roc_curve(y_true_all, y_score_all, save_path: str, title: str):
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 12

    # If multiclass, y_score_all is array of shape (N, n_classes)
    if y_score_all.ndim == 2 and y_score_all.shape[1] > 1:
        # One-vs-Rest AUC
        classes = np.unique(y_true_all)
        y_true_bin = label_binarize(y_true_all, classes=classes)
        auc = roc_auc_score(y_true_bin, y_score_all, average="macro", multi_class="ovr")
        # For plotting a single curve, take macro-average fpr/tpr
        fpr = np.linspace(0, 1, 100)
        tpr = np.zeros_like(fpr)
        # Compute interpolated TPR for each class, then average
        for idx, cls in enumerate(classes):
            fpr_i, tpr_i, _ = roc_curve(y_true_bin[:, idx], y_score_all[:, idx])
            tpr_interp = np.interp(fpr, fpr_i, tpr_i)
            tpr += tpr_interp
        tpr /= len(classes)
    else:
        # Binary case: y_score_all is shape (N,)
        fpr, tpr, _ = roc_curve(y_true_all, y_score_all)
        auc = roc_auc_score(y_true_all, y_score_all)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.3f}", color="blue")
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")
    ax.set(
        xlim=[0.0, 1.0],
        ylim=[0.0, 1.0],
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title=title,
    )
    ax.legend(loc="lower right", frameon=False)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)

def plot_accuracy_bar(mod_clf_names, accuracies, save_path: str, title: str):
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 12

    fig, ax = plt.subplots(figsize=(10, 6))
    x_pos = np.arange(len(mod_clf_names))
    ax.bar(x_pos, accuracies, color=plt.get_cmap("Pastel1").colors)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(mod_clf_names, rotation=45, ha="right", fontweight="bold")
    ax.set_ylabel("Mean Accuracy", fontweight="bold")
    ax.set_title(title, fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)

# ─── Main LOSO Evaluation ─────────────────────────────────────────────────

def main():
    cfg         = load_config()
    feat_dir    = cfg['features']['output_dir']
    sel_dir     = os.path.join(os.path.dirname(feat_dir), "selected_features")
    modalities  = cfg['modalities']
    model_types = ['xgb', 'rf', 'et', 'lgbm', 'cat']

    loso_fig_dir = os.path.join(cfg['figures']['output_dir'], "loso")
    loso_csv_dir = cfg['summary']['output_dir']
    os.makedirs(loso_fig_dir, exist_ok=True)
    os.makedirs(loso_csv_dir, exist_ok=True)

    records = []
    bar_labels = []
    bar_accuracies = []

    for mod in modalities:
        # Load full-feature CSV
        csv_path = os.path.join(feat_dir, f"{mod}_features.csv")
        df_full  = pd.read_csv(csv_path)
        df_full.columns = [c.strip() for c in df_full.columns]

        # Extract subject IDs
        if 'segment_id' in df_full.columns:
            df_full['subject'] = df_full['segment_id'].apply(extract_subject_id)
        elif 'Segment_ID' in df_full.columns:
            df_full['subject'] = df_full['Segment_ID'].apply(extract_subject_id)
        else:
            raise KeyError(f"No 'segment_id' column in {csv_path}")

        top_feats = get_topk_features(mod, sel_dir, top_k=20)
        df_feats = df_full[top_feats + ['subject', 'label']]

        subjects = sorted(df_feats['subject'].unique())

        for clf_name in model_types:
            cms = np.zeros((len(np.unique(df_feats['label'])),) * 2, dtype=int)
            y_true_all = []
            y_score_all = []
            acc_list = []
            prec_list = []
            rec_list = []
            f1_list  = []
            auc_list = []

            for test_subj in subjects:
                train_df = df_feats[df_feats['subject'] != test_subj]
                test_df  = df_feats[df_feats['subject'] == test_subj]

                # --- MODIFY: Work with DataFrames to preserve column names ---
                X_train_df = train_df[top_feats].copy()
                y_train    = train_df['label'].values
                X_test_df  = test_df[top_feats].copy()
                y_test     = test_df['label'].values

                scaler = StandardScaler().fit(X_train_df)
                X_train_s = pd.DataFrame(
                    scaler.transform(X_train_df),
                    columns=top_feats,
                    index=X_train_df.index
                )
                X_test_s = pd.DataFrame(
                    scaler.transform(X_test_df),
                    columns=top_feats,
                    index=X_test_df.index
                )

                # Instantiate classifier
                if clf_name == 'xgb':
                    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
                elif clf_name == 'rf':
                    model = RandomForestClassifier(n_estimators=100, random_state=42)
                elif clf_name == 'et':
                    model = ExtraTreesClassifier(n_estimators=100, random_state=42)
                elif clf_name == 'lgbm':
                    model = LGBMClassifier(n_estimators=100, random_state=42)
                elif clf_name == 'cat':
                    model = CatBoostClassifier(verbose=0, random_state=42)
                else:
                    raise ValueError(f"Unknown classifier: {clf_name}")

                model.fit(X_train_s, y_train)
                y_pred  = model.predict(X_test_s)
                y_proba = model.predict_proba(X_test_s)

                # Build confusion matrix for this fold
                cm_fold = confusion_matrix(y_test, y_pred, labels=np.unique(df_feats['label']))
                cms += cm_fold

                # Determine if binary or multiclass for metrics
                unique_vals = np.unique(y_test)
                if len(unique_vals) == 2:
                    # binary metrics
                    acc_fold  = accuracy_score(y_test, y_pred)
                    prec_fold = precision_score(y_test, y_pred, zero_division=0, average='binary')
                    rec_fold  = recall_score(y_test, y_pred, zero_division=0, average='binary')
                    f1_fold   = f1_score(y_test, y_pred, zero_division=0, average='binary')
                    auc_fold  = roc_auc_score(y_test, y_proba[:, 1])
                    y_true_all.extend(y_test.tolist())
                    y_score_all.extend(y_proba[:, 1].tolist())
                else:
                    # multiclass metrics
                    acc_fold  = accuracy_score(y_test, y_pred)
                    prec_fold = precision_score(y_test, y_pred, zero_division=0, average='macro')
                    rec_fold  = recall_score(y_test, y_pred, zero_division=0, average='macro')
                    f1_fold   = f1_score(y_test, y_pred, zero_division=0, average='macro')
                    classes   = np.unique(df_feats['label'])
                    y_test_b  = label_binarize(y_test, classes=classes)
                    auc_fold  = roc_auc_score(y_test_b, y_proba, average='macro', multi_class='ovr')
                    y_true_all.extend(y_test.tolist())
                    y_score_all.extend(y_proba.tolist())

                acc_list.append(acc_fold)
                prec_list.append(prec_fold)
                rec_list.append(rec_fold)
                f1_list.append(f1_fold)
                auc_list.append(auc_fold)

            # Aggregate across folds
            acc_mean, acc_std = np.nanmean(acc_list), np.nanstd(acc_list)
            prec_mean, prec_std = np.nanmean(prec_list), np.nanstd(prec_list)
            rec_mean, rec_std = np.nanmean(rec_list), np.nanstd(rec_list)
            f1_mean, f1_std = np.nanmean(f1_list), np.nanstd(f1_list)
            auc_vals = [v for v in auc_list if not np.isnan(v)]
            auc_mean = np.nanmean(auc_vals) if auc_vals else np.nan
            auc_std  = np.nanstd(auc_vals) if auc_vals else np.nan

            record = {
                'modality': mod,
                'classifier': clf_name,
                'accuracy_mean': acc_mean,
                'accuracy_std': acc_std,
                'precision_mean': prec_mean,
                'precision_std': prec_std,
                'recall_mean': rec_mean,
                'recall_std': rec_std,
                'f1_mean': f1_mean,
                'f1_std': f1_std,
                'roc_auc_mean': auc_mean,
                'roc_auc_std': auc_std
            }
            records.append(record)

            label = f"{mod}+{clf_name.upper()}"
            bar_labels.append(label)
            bar_accuracies.append(acc_mean)

            # Save Confusion Matrix plot
            cm_title = f"Confusion Matrix: {mod} + {clf_name.upper()}"
            cm_path = os.path.join(loso_fig_dir, f"CM_{mod}_{clf_name}.png")
            plot_confusion_matrix(cms, classes=np.unique(df_feats['label']).tolist(), title=cm_title, save_path=cm_path)

            # Save ROC Curve plot
            roc_title = f"ROC Curve: {mod} + {clf_name.upper()}"
            roc_path = os.path.join(loso_fig_dir, f"ROC_{mod}_{clf_name}.png")
            y_true_arr = np.array(y_true_all)
            y_score_arr = np.array(y_score_all)
            plot_roc_curve(y_true_arr, y_score_arr, save_path=roc_path, title=roc_title)

    # Save aggregate CSV
    df_metrics = pd.DataFrame(records)
    csv_out = os.path.join(loso_csv_dir, "LOSO_metrics_summary.csv")
    df_metrics.to_csv(csv_out, index=False)
    print(f"Saved LOSO metrics summary ➔ {csv_out}")

    # Save accuracy bar plot
    bar_path = os.path.join(loso_fig_dir, "Accuracy_barplot_loso.png")
    plot_accuracy_bar(bar_labels, bar_accuracies, save_path=bar_path,
                      title="Mean Accuracy by Modality+Classifier (LOSO)")
    print(f"Saved LOSO accuracy bar plot ➔ {bar_path}")

if __name__ == "__main__":
    t0 = time.perf_counter()
    main()
    print(f"Total LOSO time: {(time.perf_counter() - t0)/60:.2f} minutes")
