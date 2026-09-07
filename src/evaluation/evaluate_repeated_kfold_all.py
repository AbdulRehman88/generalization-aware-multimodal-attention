import os
import yaml
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def load_config(path="configs/config.yaml"):
    cfg = yaml.safe_load(open(path, encoding='utf-8'))
    features_dir = cfg['features']['output_dir']
    models_dir   = cfg['models']['output_dir']
    summary_dir  = cfg['summary']['output_dir']
    return features_dir, models_dir, summary_dir

def load_data(feature_file):
    df = pd.read_csv(feature_file)
    X = df.drop(columns=['segment_id', 'label'])
    y = df['label']
    return X, y

def evaluate_repeated_kfold(model, X, y, n_splits=5, n_repeats=3):
    rskf = RepeatedStratifiedKFold(n_splits=n_splits,
                                   n_repeats=n_repeats,
                                   random_state=42)
    metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
    for train_idx, test_idx in rskf.split(X, y):
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds = model.predict(X.iloc[test_idx])
        metrics['accuracy'].append(accuracy_score(y.iloc[test_idx], preds))
        metrics['precision'].append(
            precision_score(y.iloc[test_idx], preds, average='macro'))
        metrics['recall'].append(
            recall_score(y.iloc[test_idx], preds, average='macro'))
        metrics['f1'].append(
            f1_score(y.iloc[test_idx], preds, average='macro'))
    return {k: np.nanmean(v) for k, v in metrics.items()}

if __name__ == '__main__':
    features_dir, models_dir, summary_dir = load_config()

    # Prepare evaluation directory
    eval_dir = os.path.join(summary_dir, 'repeated_stratified_kfold')
    fig_dir  = os.path.join(eval_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)

    # Discover modalities
    feat_files = glob(os.path.join(features_dir, '*_features.csv'))
    modalities = [os.path.basename(f).replace('_features.csv', '')
                  for f in feat_files]

    # Discover & filter classifiers
    model_files = glob(os.path.join(models_dir, '*.pkl'))
    suffixes = {os.path.splitext(os.path.basename(f))[0].split('_')[-1]
                for f in model_files}
    allowed = {'xgb', 'rf', 'et', 'lgbm', 'cat'}
    classifiers = sorted(suffixes & allowed)

    total = len(modalities) * len(classifiers)
    print(f"Running Repeated Stratified K-Fold for "
          f"{len(modalities)} modalities × {len(classifiers)} classifiers = "
          f"{total} runs")

    # Run evaluations
    records = []
    cnt = 0
    for mod in modalities:
        X, y = load_data(os.path.join(features_dir, f"{mod}_features.csv"))
        for clf in classifiers:
            cnt += 1
            print(f"[{cnt}/{total}] {mod}_{clf}")
            model_path = os.path.join(models_dir, f"{mod}_{clf}.pkl")
            if not os.path.exists(model_path):
                print("  → Model missing, skip")
                continue
            model = joblib.load(model_path)
            scores = evaluate_repeated_kfold(model, X, y)
            scores.update({
                'modality': mod,
                'classifier': clf,
                'method': 'repeated_stratified_kfold'
            })
            records.append(scores)

    # Save CSV
    df = pd.DataFrame(records)
    csv_path = os.path.join(eval_dir, 'repeated_stratified_kfold_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"Saved results to {csv_path}")

    # Plot each metric
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    for metric in metrics:
        pivot = df.pivot(index='modality', columns='classifier', values=metric)
        ax = pivot.plot(kind='bar', figsize=(8, 5), title=metric.capitalize())
        ax.set_ylabel(metric.capitalize())
        ax.legend(title='Classifier', bbox_to_anchor=(1,1))
        plt.tight_layout()
        fig_path = os.path.join(fig_dir, f"{metric}.png")
        plt.savefig(fig_path)
        plt.close()
        print(f"Saved plot {fig_path}")
