#!/usr/bin/env python3
import os
import yaml
import joblib
import numpy as np
import pandas as pd
from glob import glob
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.preprocessing import label_binarize

def load_config():
    # locate project root two levels up
    root  = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    cfg_p = os.path.join(root, 'configs', 'config.yaml')
    with open(cfg_p, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    return (
        cfg['features']['output_dir'],
        cfg['models']['output_dir'],
        cfg['summary']['output_dir']
    )

def extend_results():
    feat_dir, model_dir, summary_dir = load_config()

    # discover modalities
    modalities = [
        os.path.basename(f).replace('_features.csv','')
        for f in glob(os.path.join(feat_dir, '*_features.csv'))
    ]
    # discover classifiers
    classifiers = sorted({
        os.path.splitext(os.path.basename(p))[0].split('_')[-1]
        for p in glob(os.path.join(model_dir, '*.pkl'))
    } & {'xgb','rf','et','lgbm','cat'})

    # infer class labels from the first features file
    first_df = pd.read_csv(glob(os.path.join(feat_dir, '*_features.csv'))[0])
    classes   = np.unique(first_df['label'])
    class_names = ['Low Attention','Mid Attention','High Attention']

    methods = ['stratified_kfold','repeated_kfold','shuffle_split','bootstrap_ci']
    for method in methods:
        out_dir = os.path.join(summary_dir, method)
        for mod in modalities:
            # load raw data
            df = pd.read_csv(os.path.join(feat_dir, f"{mod}_features.csv"))
            X  = df.drop(columns=['segment_id','label'])
            y  = df['label']
            y_bin = label_binarize(y, classes=classes)

            for clf in classifiers:
                pkl_path = os.path.join(model_dir, f"{mod}_{clf}.pkl")
                npz_path = os.path.join(out_dir,      f"roc_{mod}_{clf}.npz")
                if not os.path.exists(pkl_path) or not os.path.exists(npz_path):
                    continue

                # load existing .npz and the model
                data  = dict(np.load(npz_path, allow_pickle=True))
                model = joblib.load(pkl_path)

                # align features
                if hasattr(model, 'feature_names_in_'):
                    X_mod = X[model.feature_names_in_]
                else:
                    X_mod = X

                # compute per-class ROC & AUC on full dataset
                proba = model.predict_proba(X_mod)
                for i, cname in enumerate(class_names):
                    fpr_i, tpr_i, _ = roc_curve(y_bin[:, i], proba[:, i])
                    auc_i = auc(fpr_i, tpr_i)
                    data[f"fpr_{i}"] = fpr_i
                    data[f"tpr_{i}"] = tpr_i
                    data[f"auc_{i}"] = auc_i

                # compute confusion matrix on full dataset
                preds = model.predict(X_mod)
                cm = confusion_matrix(
                    y.map({0:0,1:1,2:2}),
                    preds,
                    labels=classes
                )
                data['cm'] = cm

                # overwrite the .npz with extended data
                np.savez(npz_path, **data)

                print(f"Extended {method}: {mod}_{clf}.npz (added per-class ROC & cm)")

    print("Extension complete: all .npz files now include fpr_i/tpr_i/auc_i & cm.")

if __name__ == "__main__":
    extend_results()
