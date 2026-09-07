# File: src/utils/inference.py
import os
import joblib
import pandas as pd

def detect_modality(file_paths):
    """
    Identify the modality (ECG, EEG, or Pupil) based on file names.
    """
    for m in ['ECG','EEG','Pupil']:
        if any(m.lower() in os.path.basename(fp).lower() for fp in file_paths):
            return m
    raise ValueError("Cannot detect modality from provided file paths")


def predict(file_paths, config, top_k=20, model_name='xgb'):
    """
    Given a list of processed segment CSV file paths (all from the same modality),
    detect modality, load the SHAP-selected feature list and corresponding trained model,
    and return predictions for each segment.

    Returns:
        List of predicted labels (0,1,2) in the same order as file_paths.
    """
    # 1) Detect modality
    mod = detect_modality(file_paths)

    # 2) Load SHAP-selected feature names
    sel_file = os.path.join(
        os.path.dirname(config['features']['output_dir']),
        'selected_features',
        f"{mod}_top{top_k}_features.txt"
    )
    if not os.path.exists(sel_file):
        raise FileNotFoundError(f"Selected features file not found: {sel_file}")
    with open(sel_file) as f:
        features = [line.strip() for line in f if line.strip()]

    # 3) Load the trained model
    model_path = os.path.join(
        config['models']['output_dir'],
        f"{mod}_{model_name}.pkl"
    )
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model = joblib.load(model_path)

    # 4) Build feature matrix
    X_list = []
    for fp in file_paths:
        df = pd.read_csv(fp)
        X_list.append(df[features])
    X_all = pd.concat(X_list, ignore_index=True)

    # 5) Predict
    preds = model.predict(X_all)
    return preds
