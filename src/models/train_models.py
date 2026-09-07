# =============================================================================
# train_models.py
# =============================================================================
# Model Training Module for Attention Detection System
#
# This module trains multiple classifiers for each modality using
# SHAP-selected features. It supports:
#   - Single top_k value (original behavior)
#   - Multiple top_k values for ablation study
#   - Organized model output directories per top_k
#
# Classifiers: XGBoost, RandomForest, ExtraTrees, LightGBM, CatBoost
#
# Author: Sanjar / Dr. Abdul Rehman Lab
# Date: November 2025
# Project: ETRI Attention Detection System
#
# IMPORTANT: Original naming conventions preserved for compatibility
# =============================================================================

import os
import yaml
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, List, Optional, Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


# =============================================================================
# CLASSIFIER DEFINITIONS
# =============================================================================

def get_classifiers(random_state: int = 42) -> Dict[str, Any]:
    """
    Get dictionary of classifiers with consistent random states.

    Args:
        random_state: Random seed for reproducibility

    Returns:
        Dictionary of classifier name -> classifier instance
    """
    return {
        'xgb': XGBClassifier(
            use_label_encoder=False,
            eval_metric='mlogloss',
            random_state=random_state,
            n_estimators=100,
            max_depth=6,
            verbosity=0
        ),
        'rf': RandomForestClassifier(
            n_estimators=100,
            random_state=random_state,
            n_jobs=-1
        ),
        'et': ExtraTreesClassifier(
            n_estimators=100,
            random_state=random_state,
            n_jobs=-1
        ),
        'lgbm': LGBMClassifier(
            n_estimators=100,
            random_state=random_state,
            verbose=-1
        ),
        'cat': CatBoostClassifier(
            iterations=100,
            random_state=random_state,
            verbose=0
        )
    }


# Legacy classifier dict for backward compatibility
CLASSIFIERS = {
    'xgb': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'),
    'rf': RandomForestClassifier(),
    'et': ExtraTreesClassifier(),
    'lgbm': LGBMClassifier(),
    'cat': CatBoostClassifier(verbose=0)
}


# =============================================================================
# CONFIGURATION LOADING
# =============================================================================

def load_config(cfg_path: Optional[str] = None) -> dict:
    """
    Load configuration from config.yaml

    Args:
        cfg_path: Optional path to config file

    Returns:
        Configuration dictionary
    """
    if cfg_path is None:
        # Try multiple possible locations
        possible_paths = [
            "configs/config.yaml",
            "../configs/config.yaml",
            "../../configs/config.yaml",
            Path(__file__).resolve().parents[2] / 'configs' / 'config.yaml'
        ]
        for p in possible_paths:
            if Path(p).exists():
                cfg_path = str(p)
                break
        else:
            raise FileNotFoundError("Could not find config.yaml")

    with open(cfg_path, encoding='utf-8') as f:
        return yaml.safe_load(f)


# =============================================================================
# FEATURE LOADING
# =============================================================================

def load_selected_features(mod: str, cfg: dict, top_k: Union[int, str]) -> List[str]:
    """
    Load selected feature names from text file.

    Args:
        mod: Modality name (e.g., 'EEG', 'ECG_EEG_Pupil')
        cfg: Configuration dictionary
        top_k: Number of features or 'all'

    Returns:
        List of feature names
    """
    suffix = 'all' if top_k == 'all' else str(top_k)
    sel_path = os.path.join(
        os.path.dirname(cfg['features']['output_dir']),
        'selected_features',
        f"{mod}_top{suffix}_features.txt"
    )

    if not os.path.exists(sel_path):
        raise FileNotFoundError(f"Selected features file not found: {sel_path}")

    with open(sel_path, encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def train_for_modality(mod: str, cfg: dict, top_k: Union[int, str] = 20,
                       output_subdir: bool = False, verbose: bool = True) -> Dict[str, dict]:
    """
    Train all classifiers for a single modality.

    Args:
        mod: Modality name
        cfg: Configuration dictionary
        top_k: Number of features to use or 'all'
        output_subdir: If True, save models in top_k subdirectory
        verbose: Whether to print progress

    Returns:
        Dictionary of classifier results with metrics
    """
    feat_dir = cfg['features']['output_dir']
    feat_file = os.path.join(feat_dir, f"{mod}_features.csv")

    if not os.path.exists(feat_file):
        if verbose:
            print(f"  [WARNING] Feature file not found: {feat_file}")
        return {}

    # Load features
    df = pd.read_csv(feat_file)
    sel_feats = load_selected_features(mod, cfg, top_k)

    # Verify all selected features exist in DataFrame
    missing_feats = [f for f in sel_feats if f not in df.columns]
    if missing_feats:
        if verbose:
            print(f"  [WARNING] Missing features: {missing_feats[:5]}...")
        sel_feats = [f for f in sel_feats if f in df.columns]

    if not sel_feats:
        if verbose:
            print(f"  [ERROR] No valid features for {mod}")
        return {}

    X = df[sel_feats]
    y = df['label']

    if verbose:
        suffix = 'all' if top_k == 'all' else str(top_k)
        print(f"  Features: {len(sel_feats)} (top_{suffix})")
        print(f"  Samples: {len(X)}")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Output directory
    if output_subdir:
        suffix = 'all' if top_k == 'all' else str(top_k)
        out_dir = os.path.join(cfg['models']['output_dir'], f"top_{suffix}")
    else:
        out_dir = cfg['models']['output_dir']
    os.makedirs(out_dir, exist_ok=True)

    # Train each classifier
    results = {}
    classifiers = get_classifiers(random_state=42)

    for name, clf in classifiers.items():
        try:
            # Train
            clf.fit(X_train, y_train)

            # Predict
            preds = clf.predict(X_test)

            # Calculate metrics
            acc = accuracy_score(y_test, preds)
            prec = precision_score(y_test, preds, average='macro', zero_division=0)
            rec = recall_score(y_test, preds, average='macro', zero_division=0)
            f1 = f1_score(y_test, preds, average='macro', zero_division=0)

            results[name] = {
                'accuracy': acc,
                'precision': prec,
                'recall': rec,
                'f1': f1
            }

            if verbose:
                print(f"    {name.upper()}: Acc={acc:.4f}, F1={f1:.4f}")

            # Save model
            model_path = os.path.join(out_dir, f"{mod}_{name}.pkl")
            joblib.dump(clf, model_path)

        except Exception as e:
            if verbose:
                print(f"    {name.upper()}: [ERROR] {e}")
            results[name] = {'error': str(e)}

    return results


def train_all_modalities(cfg: dict, top_k: Union[int, str] = 20,
                         output_subdir: bool = False, verbose: bool = True) -> Dict[str, Dict[str, dict]]:
    """
    Train models for all modalities in config.

    Args:
        cfg: Configuration dictionary
        top_k: Number of features or 'all'
        output_subdir: If True, save models in top_k subdirectory
        verbose: Whether to print progress

    Returns:
        Nested dictionary: modality -> classifier -> metrics
    """
    all_results = {}

    for mod in cfg['modalities']:
        if verbose:
            suffix = 'all' if top_k == 'all' else str(top_k)
            print(f"\n[{mod}] Training with top_{suffix} features...")

        try:
            results = train_for_modality(mod, cfg, top_k, output_subdir, verbose)
            all_results[mod] = results
        except Exception as e:
            if verbose:
                print(f"  [ERROR] {e}")
            all_results[mod] = {'error': str(e)}

    return all_results


# =============================================================================
# ABLATION STUDY TRAINING
# =============================================================================

def train_ablation_study(
        top_k_values: List[Union[int, str]] = [10, 20, 30, 40, 50, 60, 70, 80, 'all'],
        modalities: Optional[List[str]] = None,
        save_summary: bool = True,
        verbose: bool = True
) -> pd.DataFrame:
    """
    Train models for all top_k values (ablation study).

    This function:
    1. Trains all classifiers for each modality and top_k value
    2. Organizes models in subdirectories by top_k
    3. Saves a summary CSV with all results

    Args:
        top_k_values: List of top_k values to train
        modalities: List of modalities (None = all from config)
        save_summary: Whether to save summary CSV
        verbose: Whether to print progress

    Returns:
        DataFrame with columns: [modality, classifier, top_k, accuracy, precision, recall, f1]
    """
    cfg = load_config()

    if modalities is None:
        modalities = cfg['modalities']

    if verbose:
        print("=" * 70)
        print("ABLATION STUDY: MODEL TRAINING")
        print("=" * 70)
        print(f"Top-K values: {top_k_values}")
        print(f"Modalities: {modalities}")
        print(f"Classifiers: {list(get_classifiers().keys())}")
        print("=" * 70)

    # Collect all results
    all_records = []

    for top_k in top_k_values:
        suffix = 'all' if top_k == 'all' else str(top_k)

        if verbose:
            print(f"\n{'=' * 70}")
            print(f"TRAINING WITH TOP_{suffix} FEATURES")
            print(f"{'=' * 70}")

        for mod in modalities:
            if verbose:
                print(f"\n[{mod}] Training...")

            try:
                results = train_for_modality(
                    mod, cfg, top_k,
                    output_subdir=True,  # Organize by top_k
                    verbose=verbose
                )

                # Record results
                for clf_name, metrics in results.items():
                    if 'error' not in metrics:
                        all_records.append({
                            'modality': mod,
                            'classifier': clf_name,
                            'top_k': suffix,
                            'accuracy': metrics['accuracy'],
                            'precision': metrics['precision'],
                            'recall': metrics['recall'],
                            'f1': metrics['f1']
                        })

            except Exception as e:
                if verbose:
                    print(f"  [ERROR] {e}")

    # Create summary DataFrame
    df_summary = pd.DataFrame(all_records)

    # Save summary
    if save_summary and not df_summary.empty:
        summary_dir = cfg['summary']['output_dir']
        os.makedirs(summary_dir, exist_ok=True)
        summary_path = os.path.join(summary_dir, 'ablation_training_summary.csv')
        df_summary.to_csv(summary_path, index=False)

        if verbose:
            print(f"\n{'=' * 70}")
            print("ABLATION TRAINING COMPLETE")
            print(f"{'=' * 70}")
            print(f"Summary saved: {summary_path}")
            print(f"Total configurations trained: {len(df_summary)}")

    return df_summary


# =============================================================================
# MAIN FUNCTION (Backward Compatible)
# =============================================================================

def main(top_k: Union[int, str, List[Union[int, str]]] = 20):
    """
    Main entry point for model training.

    Args:
        top_k: Single value for original behavior, or list for ablation study
    """
    cfg = load_config()

    if isinstance(top_k, list):
        # Ablation study
        train_ablation_study(top_k_values=top_k)
    else:
        # Original behavior
        print("=" * 70)
        print(f"MODEL TRAINING (top_k={top_k})")
        print("=" * 70)

        for mod in cfg['modalities']:
            print(f"\nTraining models for {mod}")
            train_for_modality(mod, cfg, top_k=top_k, verbose=True)

        print("\n" + "=" * 70)
        print("TRAINING COMPLETE")
        print("=" * 70)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_model_path(modality: str, classifier: str, top_k: Union[int, str] = 20,
                   cfg: Optional[dict] = None) -> str:
    """
    Get the path to a trained model file.

    Args:
        modality: Modality name
        classifier: Classifier name (xgb, rf, et, lgbm, cat)
        top_k: Top-K value used for training
        cfg: Configuration dict (loads if not provided)

    Returns:
        Path to model .pkl file
    """
    if cfg is None:
        cfg = load_config()

    suffix = 'all' if top_k == 'all' else str(top_k)
    model_dir = os.path.join(cfg['models']['output_dir'], f"top_{suffix}")

    return os.path.join(model_dir, f"{modality}_{classifier}.pkl")


def load_model(modality: str, classifier: str, top_k: Union[int, str] = 20,
               cfg: Optional[dict] = None):
    """
    Load a trained model.

    Args:
        modality: Modality name
        classifier: Classifier name
        top_k: Top-K value
        cfg: Configuration dict

    Returns:
        Loaded model object
    """
    model_path = get_model_path(modality, classifier, top_k, cfg)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    return joblib.load(model_path)


def list_trained_models(cfg: Optional[dict] = None) -> pd.DataFrame:
    """
    List all trained models with their configurations.

    Returns:
        DataFrame with columns: [modality, classifier, top_k, path]
    """
    if cfg is None:
        cfg = load_config()

    model_base = cfg['models']['output_dir']
    records = []

    # Check main directory
    for f in Path(model_base).glob("*.pkl"):
        parts = f.stem.split('_')
        clf = parts[-1]
        mod = '_'.join(parts[:-1])
        records.append({
            'modality': mod,
            'classifier': clf,
            'top_k': 'default',
            'path': str(f)
        })

    # Check subdirectories (top_X folders)
    for subdir in Path(model_base).glob("top_*"):
        if subdir.is_dir():
            top_k = subdir.name.replace('top_', '')
            for f in subdir.glob("*.pkl"):
                parts = f.stem.split('_')
                clf = parts[-1]
                mod = '_'.join(parts[:-1])
                records.append({
                    'modality': mod,
                    'classifier': clf,
                    'top_k': top_k,
                    'path': str(f)
                })

    return pd.DataFrame(records)


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Train models for Attention Detection System'
    )
    parser.add_argument(
        '--top_k',
        type=str,
        default='20',
        help='Top-K value(s). Comma-separated for multiple (e.g., "10,20,30,all"). Default: 20'
    )
    parser.add_argument(
        '--ablation',
        action='store_true',
        help='Run full ablation study with default top_k values [10,20,30,40,50,60,70,80,all]'
    )
    parser.add_argument(
        '--modality',
        type=str,
        default=None,
        help='Train only specific modality (e.g., "ECG_EEG_Pupil")'
    )

    args = parser.parse_args()

    if args.ablation:
        # Full ablation study
        train_ablation_study()
    else:
        # Parse top_k values
        top_k_values = []
        for v in args.top_k.split(','):
            v = v.strip()
            if v.lower() == 'all':
                top_k_values.append('all')
            else:
                try:
                    top_k_values.append(int(v))
                except ValueError:
                    print(f"Warning: Invalid top_k value '{v}', skipping.")

        if not top_k_values:
            top_k_values = [20]

        if len(top_k_values) == 1:
            # Single value
            main(top_k=top_k_values[0])
        else:
            # Multiple values
            train_ablation_study(top_k_values=top_k_values)