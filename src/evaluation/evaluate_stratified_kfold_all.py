import os
import yaml
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score,
    roc_curve, auc
)

# -----------------------------------------------------------------------------
# CONFIG & DATA LOADING
# -----------------------------------------------------------------------------
def load_config(path="configs/config.yaml"):
    cfg = yaml.safe_load(open(path, encoding='utf-8'))
    feat_dir  = cfg['features']['output_dir']
    model_dir = cfg['models']['output_dir']
    sum_dir   = cfg['summary']['output_dir']
    return feat_dir, model_dir, sum_dir

def load_data(csv_path):
    df = pd.read_csv(csv_path)
    X = df.drop(columns=['segment_id','label'])
    y = df['label']
    return X, y

# -----------------------------------------------------------------------------
# EVALUATION FUNCTION (no ROC-AUC here)
# -----------------------------------------------------------------------------
def evaluate_stratified_kfold(model, X, y, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    m = {'accuracy':[], 'precision':[], 'recall':[], 'f1':[]}
    for tr, tt in skf.split(X,y):
        model.fit(X.iloc[tr], y.iloc[tr])
        preds = model.predict(X.iloc[tt])
        m['accuracy'].append(accuracy_score(y.iloc[tt], preds))
        m['precision'].append(precision_score(y.iloc[tt], preds, average='macro'))
        m['recall'].append(recall_score(y.iloc[tt], preds, average='macro'))
        m['f1'].append(f1_score(y.iloc[tt], preds, average='macro'))
    return {k: np.nanmean(v) for k,v in m.items()}

# -----------------------------------------------------------------------------
# MAIN SCRIPT
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    feat_dir, model_dir, sum_dir = load_config()
    eval_dir = os.path.join(sum_dir, "stratified_kfold")
    fig_dir  = os.path.join(eval_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # discover modalities
    feat_files = glob(os.path.join(feat_dir, "*_features.csv"))
    modalities = [os.path.basename(f).replace("_features.csv","") for f in feat_files]

    # discover & filter classifiers
    model_files = glob(os.path.join(model_dir, "*.pkl"))
    suffixes = {os.path.splitext(os.path.basename(m))[0].split("_")[-1] for m in model_files}
    allowed = {"xgb","rf","et","lgbm","cat"}
    classifiers = sorted(suffixes & allowed)

    total = len(modalities) * len(classifiers)
    print(f"Running Stratified K-Fold: {len(modalities)} modalities × {len(classifiers)} classifiers = {total} runs")

    # evaluation loop with progress bar
    records = []
    pbar = tqdm(total=total, desc="Stratified K-Fold", unit="run")
    for mod in modalities:
        X, y = load_data(os.path.join(feat_dir, f"{mod}_features.csv"))
        for clf in classifiers:
            model_path = os.path.join(model_dir, f"{mod}_{clf}.pkl")
            if not os.path.exists(model_path):
                pbar.update(1)
                continue
            model = joblib.load(model_path)
            scores = evaluate_stratified_kfold(model, X, y)
            scores.update(modality=mod, classifier=clf)
            records.append(scores)
            pbar.update(1)
    pbar.close()

    # save CSV
    df = pd.DataFrame(records)
    csv_path = os.path.join(eval_dir, "stratified_kfold_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved results → {csv_path}")

    # set global font
    plt.rc('font', family='Times New Roman', size=12)

    # bar plots: accuracy, precision, recall, f1
    pastel = plt.get_cmap("Pastel1").colors
    for metric in ["accuracy","precision","recall","f1"]:
        pivot = df.pivot(index="modality", columns="classifier", values=metric)
        ax = pivot.plot(
            kind="bar", figsize=(8,5),
            color=pastel, edgecolor="black"
        )
        ax.set_title(metric.capitalize(), fontweight="bold", color="black")
        ax.set_xlabel("Modality", fontweight="bold", color="black")
        ax.set_ylabel(metric.capitalize(), fontweight="bold", color="black")
        ax.tick_params(labelsize=12, colors="black")
        leg = ax.legend(
            title="Classifier",
            prop={"size":12,"weight":"bold","family":"Times New Roman"},
            frameon=False
        )
        for text in leg.get_texts():
            text.set_fontweight("bold")
            text.set_color("black")
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, f"{metric}.png"), dpi=300, bbox_inches="tight")
        plt.close()

    # multiclass ROC curves (micro-average)
    from sklearn.preprocessing import label_binarize
    from sklearn.model_selection import StratifiedKFold
    for mod in modalities:
        X, y = load_data(os.path.join(feat_dir, f"{mod}_features.csv"))
        y_bin = label_binarize(y, classes=np.unique(y))
        for clf in classifiers:
            model_path = os.path.join(model_dir, f"{mod}_{clf}.pkl")
            if not os.path.exists(model_path):
                continue
            model = joblib.load(model_path)
            # cross-validated probabilities
            y_score = cross_val_predict(
                model, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=42),
                method="predict_proba"
            )
            # micro-average ROC
            fpr, tpr, _ = roc_curve(y_bin.ravel(), y_score.ravel())
            roc_auc = auc(fpr, tpr)
            plt.figure(figsize=(8,6))
            plt.plot(fpr, tpr, lw=2, label=f"{mod}_{clf} (AUC={roc_auc:.2f})")
            plt.plot([0,1],[0,1], '--', color="gray", lw=1)
            plt.title("ROC Curve", fontweight="bold", color="black")
            plt.xlabel("False Positive Rate", fontweight="bold", color="black")
            plt.ylabel("True Positive Rate", fontweight="bold", color="black")
            plt.tick_params(labelsize=12, colors="black")
            leg = plt.legend(
                prop={"size":12,"weight":"bold","family":"Times New Roman"},
                loc="lower right", frameon=False
            )
            for text in leg.get_texts():
                text.set_fontweight("bold")
                text.set_color("black")
            plt.tight_layout()
            plt.savefig(os.path.join(fig_dir, f"roc_{mod}_{clf}.png"),
                        dpi=300, bbox_inches="tight")
            plt.close()
    print(f"Saved ROC curves → {fig_dir}")
