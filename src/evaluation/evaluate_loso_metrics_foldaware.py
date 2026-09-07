# evaluate_loso_metrics_foldaware.py
# Fully optimized LOSO script with fold-aware SHAP feature selection and subject-level normalization
import os
import glob
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, roc_curve
)
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import yaml

def load_config():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "..", "..", "configs", "config.yaml")
    config_path = os.path.normpath(config_path)
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)

def extract_subject_id(segment_id: str) -> str:
    part = segment_id.split(".")[1]
    return part.split("_")[0]

def get_fold_shap_features(modality: str, subject_id: str, sel_dir: str, top_k: int = 20):
    fold_file = os.path.join(sel_dir, f"{modality}_{subject_id}_top{top_k}_features.txt")
    global_file = os.path.join(sel_dir, f"{modality}_top{top_k}_features.txt")

    if os.path.isfile(fold_file):
        with open(fold_file, encoding="utf-8") as f:
            return f.read().splitlines()[:top_k]
    elif os.path.isfile(global_file):
        with open(global_file, encoding="utf-8") as f:
            print(f"[WARNING] Using global top-k features for {modality}, subject {subject_id}")
            return f.read().splitlines()[:top_k]
    else:
        raise FileNotFoundError(f"No SHAP feature file for '{modality}' or '{modality}_{subject_id}'")

def plot_confusion_matrix(cm, classes, title, save_path):
    plt.rcParams["font.family"] = "Times New Roman"
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(cm, cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(len(classes)), yticks=np.arange(len(classes)),
           xticklabels=classes, yticklabels=classes, ylabel="True", xlabel="Predicted", title=title)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)

def plot_roc_curve(y_true_all, y_score_all, save_path, title):
    plt.rcParams["font.family"] = "Times New Roman"
    if y_score_all.ndim == 2 and y_score_all.shape[1] > 1:
        classes = np.unique(y_true_all)
        y_true_bin = label_binarize(y_true_all, classes=classes)
        auc = roc_auc_score(y_true_bin, y_score_all, average="macro", multi_class="ovr")
        fpr = np.linspace(0, 1, 100)
        tpr = np.zeros_like(fpr)
        for i in range(len(classes)):
            fpr_i, tpr_i, _ = roc_curve(y_true_bin[:, i], y_score_all[:, i])
            tpr += np.interp(fpr, fpr_i, tpr_i)
        tpr /= len(classes)
    else:
        fpr, tpr, _ = roc_curve(y_true_all, y_score_all)
        auc = roc_auc_score(y_true_all, y_score_all)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--")
    ax.set(xlabel="FPR", ylabel="TPR", title=title)
    ax.legend(loc="lower right")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)

def main():
    cfg = load_config()
    feat_dir = cfg['features']['output_dir']
    sel_dir = os.path.join(os.path.dirname(feat_dir), "selected_features")
    modalities = cfg['modalities']
    model_types = ['xgb', 'rf', 'et', 'lgbm', 'cat']
    fig_dir = os.path.join(cfg['figures']['output_dir'], "loso_foldaware")
    csv_dir = os.path.join(cfg['summary']['output_dir'], "loso_foldaware")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)

    records = []
    for mod in modalities:
        df_path = os.path.join(feat_dir, f"{mod}_features.csv")
        df = pd.read_csv(df_path)
        df.columns = [c.strip() for c in df.columns]
        seg_col = 'segment_id' if 'segment_id' in df.columns else 'Segment_ID'
        df['subject'] = df[seg_col].apply(extract_subject_id)
        subjects = sorted(df['subject'].unique())

        for clf_name in model_types:
            accs, precs, recs, f1s, aucs = [], [], [], [], []
            cms = np.zeros((len(np.unique(df['label'])),) * 2, dtype=int)
            y_true_all, y_score_all = [], []

            for subj in subjects:
                fold_feats = get_fold_shap_features(mod, subj, sel_dir, top_k=20)
                fold_df = df[fold_feats + ['subject', 'label']]
                train = fold_df[fold_df['subject'] != subj]
                test = fold_df[fold_df['subject'] == subj]

                X_train, y_train = train[fold_feats], train['label'].values
                X_test, y_test = test[fold_feats], test['label'].values

                scaler = StandardScaler().fit(X_train)
                X_train = scaler.transform(X_train)
                X_test = scaler.transform(X_test)

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

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test)

                cms += confusion_matrix(y_test, y_pred, labels=np.unique(df['label']))
                accs.append(accuracy_score(y_test, y_pred))
                if len(np.unique(y_test)) == 2:
                    precs.append(precision_score(y_test, y_pred, average='binary', zero_division=0))
                    recs.append(recall_score(y_test, y_pred, average='binary', zero_division=0))
                    f1s.append(f1_score(y_test, y_pred, average='binary', zero_division=0))
                    aucs.append(roc_auc_score(y_test, y_proba[:, 1]))
                    y_true_all.extend(y_test.tolist())
                    y_score_all.extend(y_proba[:, 1].tolist())
                else:
                    precs.append(precision_score(y_test, y_pred, average='macro', zero_division=0))
                    recs.append(recall_score(y_test, y_pred, average='macro', zero_division=0))
                    f1s.append(f1_score(y_test, y_pred, average='macro', zero_division=0))
                    y_bin = label_binarize(y_test, classes=np.unique(df['label']))
                    aucs.append(roc_auc_score(y_bin, y_proba, average='macro', multi_class='ovr'))
                    y_true_all.extend(y_test.tolist())
                    y_score_all.extend(y_proba.tolist())

            records.append({
                'modality': mod, 'classifier': clf_name,
                'accuracy_mean': np.mean(accs), 'accuracy_std': np.std(accs),
                'precision_mean': np.mean(precs), 'precision_std': np.std(precs),
                'recall_mean': np.mean(recs), 'recall_std': np.std(recs),
                'f1_mean': np.mean(f1s), 'f1_std': np.std(f1s),
                'roc_auc_mean': np.mean(aucs), 'roc_auc_std': np.std(aucs)
            })

            plot_confusion_matrix(cms, classes=np.unique(df['label']),
                                  title=f"CM: {mod}+{clf_name}",
                                  save_path=os.path.join(fig_dir, f"CM_{mod}_{clf_name}.png"))

            plot_roc_curve(np.array(y_true_all), np.array(y_score_all),
                           save_path=os.path.join(fig_dir, f"ROC_{mod}_{clf_name}.png"),
                           title=f"ROC: {mod}+{clf_name}")

    pd.DataFrame(records).to_csv(os.path.join(csv_dir, "LOSO_metrics_summary_foldaware.csv"), index=False)
    print(f"✅ Results saved to: {csv_dir}")

if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"⏱️ Total time: {(time.time() - t0) / 60:.2f} minutes")
