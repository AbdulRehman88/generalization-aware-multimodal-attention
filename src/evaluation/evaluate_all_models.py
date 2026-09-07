#!/usr/bin/env python3
import os
import yaml
import joblib
import pandas as pd
import numpy as np
from glob import glob
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, ShuffleSplit
from sklearn.preprocessing import label_binarize
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc

def load_config():
    # locate project root (two levels up from this file)
    root    = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    cfgpath = os.path.join(root, 'configs', 'config.yaml')
    with open(cfgpath, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    return (
        cfg['features']['output_dir'],
        cfg['models']['output_dir'],
        cfg['summary']['output_dir']
    )

def load_data(csv_path):
    df = pd.read_csv(csv_path)
    return df.drop(columns=['segment_id','label']), df['label']

def compute_stratified(model, X, y):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    return _compute_common(model, X, y, skf)

def compute_repeated(model, X, y):
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
    return _compute_common(model, X, y, rskf)

def compute_shuffle(model, X, y):
    ss = ShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
    return _compute_common(model, X, y, ss)

def compute_bootstrap(model, X, y):
    rng = np.random.RandomState(42)
    accs = []
    for _ in range(100):
        idx = rng.choice(len(y), len(y), replace=True)
        model.fit(X.iloc[idx], y.iloc[idx])
        accs.append(accuracy_score(y, model.predict(X)))
    low, high = np.percentile(accs, [2.5, 97.5])
    return {
        'accuracy':       np.mean(accs),
        'precision':      np.nan,
        'recall':         np.nan,
        'f1':             np.nan,
        'roc_fpr':        np.array([]),
        'roc_tpr':        np.array([]),
        'roc_auc':        np.nan,
        'accuracy_lower': low,
        'accuracy_upper': high
    }

def _compute_common(model, X, y, splitter):
    mets = {
        'accuracy':[], 'precision':[], 'recall':[], 'f1':[],
        'roc_fpr':[], 'roc_tpr':[], 'roc_auc':[]
    }
    y_bin = label_binarize(y, classes=np.unique(y))
    for tr, tt in splitter.split(X, y):
        model.fit(X.iloc[tr], y.iloc[tr])
        preds = model.predict(X.iloc[tt])
        mets['accuracy'].append(accuracy_score(y.iloc[tt], preds))
        mets['precision'].append(precision_score(y.iloc[tt], preds, average='macro'))
        mets['recall'].append(recall_score(y.iloc[tt], preds, average='macro'))
        mets['f1'].append(f1_score(y.iloc[tt], preds, average='macro'))
        if hasattr(model, 'predict_proba'):
            prob = model.predict_proba(X.iloc[tt])
            fpr, tpr, _ = roc_curve(y_bin[tt].ravel(), prob.ravel())
            mets['roc_fpr'].append(fpr)
            mets['roc_tpr'].append(tpr)
            mets['roc_auc'].append(auc(fpr, tpr))

    # compute mean & std
    acc = np.array(mets['accuracy'])
    prec= np.array(mets['precision'])
    rec = np.array(mets['recall'])
    f1a = np.array(mets['f1'])
    aucs= np.array(mets['roc_auc']) if mets['roc_auc'] else np.array([])

    return {
        'accuracy':      acc.mean(),   'accuracy_std':   acc.std(ddof=1),
        'precision':     prec.mean(),  'precision_std':  prec.std(ddof=1),
        'recall':        rec.mean(),   'recall_std':     rec.std(ddof=1),
        'f1':            f1a.mean(),   'f1_std':         f1a.std(ddof=1),
        'roc_fpr':       np.concatenate(mets['roc_fpr']) if mets['roc_fpr'] else np.array([]),
        'roc_tpr':       np.concatenate(mets['roc_tpr']) if mets['roc_tpr'] else np.array([]),
        'roc_auc':       aucs.mean() if aucs.size else np.nan,
        'roc_auc_std':   aucs.std(ddof=1) if aucs.size else np.nan
    }

if __name__ == "__main__":
    feat_dir, model_dir, summary_dir = load_config()
    methods = {
        'stratified_kfold': compute_stratified,
        'repeated_kfold':   compute_repeated,
        'shuffle_split':    compute_shuffle,
        'bootstrap_ci':     compute_bootstrap
    }

    # discover modalities & classifiers
    mods   = [os.path.basename(f).replace('_features.csv','')
              for f in glob(os.path.join(feat_dir, '*_features.csv'))]
    clfs   = sorted({os.path.splitext(os.path.basename(m))[0].split('_')[-1]
              for m in glob(os.path.join(model_dir, '*.pkl'))} & {'xgb','rf','et','lgbm','cat'})

    for method, func in methods.items():
        out_dir = os.path.join(summary_dir, method); os.makedirs(out_dir, exist_ok=True)
        pbar = tqdm(total=len(mods)*len(clfs), desc=method, unit='run')

        for mod in mods:
            X, y = load_data(os.path.join(feat_dir, f"{mod}_features.csv"))
            for clf in clfs:
                pbar.update(1)
                path = os.path.join(model_dir, f"{mod}_{clf}.pkl")
                if not os.path.exists(path): continue
                model = joblib.load(path)
                stats = func(model, X, y)

                # write metrics CSV
                rows = []
                if method != 'bootstrap_ci':
                    for key in ['accuracy','precision','recall','f1','roc_auc']:
                        rows.append({'metric': key,          'value': stats[key]})
                        rows.append({'metric': f"{key}_std", 'value': stats[f"{key}_std"]})
                else:
                    rows.append({'metric':'accuracy',       'value': stats['accuracy']})
                    rows.append({'metric':'accuracy_lower', 'value': stats['accuracy_lower']})
                    rows.append({'metric':'accuracy_upper', 'value': stats['accuracy_upper']})

                pd.DataFrame(rows).to_csv(
                    os.path.join(out_dir, f"metrics_{mod}_{clf}.csv"),
                    index=False
                )

                # write ROC .npz
                np.savez(
                    os.path.join(out_dir, f"roc_{mod}_{clf}.npz"),
                    fpr=stats['roc_fpr'],
                    tpr=stats['roc_tpr'],
                    auc=stats['roc_auc']
                )

        pbar.close()
    print("Phase 1 complete: metrics + std saved under outputs/summary/<method>/")
