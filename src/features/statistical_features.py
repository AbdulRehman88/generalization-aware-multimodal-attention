### src/features/statistical_features.py
import os
import pandas as pd

def extract_stat_features(df):
    feats = {
        'mean_all': df.values.mean(),
        'std_all': df.values.std(),
        'max_all': df.values.max(),
        'min_all': df.values.min(),
        'median_all': pd.Series(df.values.flatten()).median()
    }
    return feats

def process_all_segments(config, modality):
    in_dir = os.path.join(config['data']['processed'], modality)
    out_dir = config['features']['output_dir']
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for f in os.listdir(in_dir):
        if not f.endswith('.csv'): continue
        df = pd.read_csv(os.path.join(in_dir, f))
        feats = extract_stat_features(df.drop(columns=['label']))
        feats['segment_id'] = f
        feats['label'] = df['label'].iloc[0]
        rows.append(feats)
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, f"{modality}_features.csv"), index=False)