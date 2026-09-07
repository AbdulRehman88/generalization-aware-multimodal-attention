#!/usr/bin/env python3
import os
import yaml
import joblib
import pandas as pd
import numpy as np
from glob import glob
from tqdm import tqdm
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score
)

def load_config():
    # __file__ is .../AttentionDetectionSystem/src/evaluation/evaluate_bootstrap_ci_all.py
    this_dir = os.path.abspath(os.path.dirname(__file__))       # .../src/evaluation
    project_root = os.path.abspath(os.path.join(this_dir, '..', '..'))  # up two → AttentionDetectionSystem
    cfg_path = os.path.join(project_root, 'configs', 'config.yaml')
    with open(cfg_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    return (
        cfg['features']['output_dir'],
        cfg['models']['output_dir'],
        cfg['summary']['output_dir']
    )

def single_bootstrap(model, X, y, B=1000, random_state=42):
    rng = np.random.RandomState(random_state)
    accs, precs, recs, f1s, aucs = [], [], [], [], []
    y_bin = label_binarize(y, classes=np.unique(y))

    for _ in range(B):
        idx = rng.choice(len(y), len(y), replace=True)
        model.fit(X.iloc[idx], y.iloc[idx])
        preds = model.predict(X)
        probs = model.predict_proba(X) if hasattr(model,'predict_proba') else None

        accs.append( accuracy_score(y, preds) )
        precs.append( precision_score(y, preds, average='macro', zero_division=0) )
        recs.append( recall_score  (y, preds, average='macro', zero_division=0) )
        f1s .append( f1_score      (y, preds, average='macro', zero_division=0) )
        if probs is not None:
            try:
                aucs.append( roc_auc_score(y, probs, average='macro', multi_class='ovr') )
            except ValueError:
                aucs.append(np.nan)
        else:
            aucs.append(np.nan)

    def summarize(arr):
        a = np.array(arr)[~np.isnan(arr)]
        return a.mean(), a.std(ddof=1), *np.percentile(a, [2.5,97.5])

    return {
        'accuracy':   summarize(accs),
        'precision':  summarize(precs),
        'recall':     summarize(recs),
        'f1':         summarize(f1s),
        'roc_auc':    summarize(aucs)
    }

if __name__ == "__main__":
    feat_dir, model_dir, summary_dir = load_config()
    out_dir = os.path.join(summary_dir, 'bootstrap_ci')
    os.makedirs(out_dir, exist_ok=True)

    feats = glob(os.path.join(feat_dir, '*_features.csv'))
    for feat in feats:
        mod = os.path.basename(feat).replace('_features.csv','')
        X = pd.read_csv(feat).drop(columns=['segment_id','label'])
        y = pd.read_csv(feat)['label']

        for pkl in glob(os.path.join(model_dir, f"{mod}_*.pkl")):
            clf = os.path.basename(pkl).split('_')[-1].replace('.pkl','')
            model = joblib.load(pkl)

            stats = single_bootstrap(model, X, y, B=1000)

            rows = []
            for metric, (mu, sd, lo, hi) in stats.items():
                rows += [
                    {'metric': metric,           'value': mu},
                    {'metric': f"{metric}_std",  'value': sd},
                    {'metric': f"{metric}_lower",'value': lo},
                    {'metric': f"{metric}_upper",'value': hi},
                ]

            df = pd.DataFrame(rows)
            csvf = os.path.join(out_dir, f"metrics_{mod}_{clf.lower()}.csv")
            df.to_csv(csvf, index=False)
            print(f"Updated bootstrap metrics → {csvf}")
