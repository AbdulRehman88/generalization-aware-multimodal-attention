# =============================================================================
# select_features.py
# =============================================================================
# SHAP-based Feature Selection for Attention Detection System
#
# This module performs SHAP (SHapley Additive exPlanations) based feature
# selection using XGBoost classifier. It supports:
#   - Single top_k value (original behavior)
#   - Multiple top_k values for ablation study
#   - "all" option to use all features
#
# Output files are saved with original naming convention:
#   <modality>_top<k>_features.txt
#
# Author: Sanjar / Dr. Abdul Rehman Lab
# Date: November 2025
# Project: ETRI Attention Detection System
#
# IMPORTANT: Original naming conventions preserved for compatibility
# =============================================================================

import os
import yaml
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
import shap
from pathlib import Path
from typing import Union, List, Optional
import warnings

# Suppress XGBoost warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


# =============================================================================
# CONFIGURATION LOADING
# =============================================================================

def load_config(cfg_path: Optional[Path] = None) -> dict:
    """
    Load configuration from config.yaml

    Args:
        cfg_path: Optional path to config file. If None, auto-detects.

    Returns:
        Configuration dictionary
    """
    if cfg_path is None:
        # Auto-detect config path (two levels up from this file)
        PROJECT_ROOT = Path(__file__).resolve().parents[2]
        cfg_path = PROJECT_ROOT / 'configs' / 'config.yaml'

    if not cfg_path.exists():
        # Fallback paths
        fallbacks = [
            Path('configs/config.yaml'),
            Path('../configs/config.yaml'),
            Path('../../configs/config.yaml')
        ]
        for fb in fallbacks:
            if fb.exists():
                cfg_path = fb
                break
        else:
            raise FileNotFoundError(f"Could not find config.yaml. Tried: {cfg_path}, {fallbacks}")

    with open(cfg_path, encoding='utf-8') as f:
        return yaml.safe_load(f)


# =============================================================================
# SHAP FEATURE IMPORTANCE COMPUTATION
# =============================================================================

def compute_shap_importance(X: pd.DataFrame, y: pd.Series, random_state: int = 42) -> pd.DataFrame:
    """
    Compute SHAP-based feature importance using XGBoost.

    Args:
        X: Feature DataFrame
        y: Label Series
        random_state: Random seed for reproducibility

    Returns:
        DataFrame with columns ['feature', 'importance'] sorted by importance (descending)
    """
    # Train XGBoost model
    model = XGBClassifier(
        use_label_encoder=False,
        eval_metric='mlogloss',
        random_state=random_state,
        n_estimators=100,
        max_depth=6,
        verbosity=0
    )
    model.fit(X, y)

    # Compute SHAP values
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X)

    # Normalize to (n_samples, n_features, n_classes) shape
    if isinstance(shap_vals, list):
        # Multi-class: list of arrays, one per class
        abs_vals = [np.abs(arr) for arr in shap_vals]
        arr = np.stack(abs_vals, axis=2)
    elif isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3:
        # Already 3D
        arr = np.abs(shap_vals)
    elif isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 2:
        # Binary classification: 2D array
        arr = np.abs(shap_vals)[..., np.newaxis]
    else:
        raise ValueError(f"Unexpected SHAP output type: {type(shap_vals)}")

    # Compute mean absolute importance over samples and classes
    # Sum across classes first, then mean across samples
    mean_importance = np.mean(np.sum(arr, axis=2), axis=0)

    # Create importance DataFrame
    importance_df = pd.DataFrame({
        'feature': X.columns.tolist(),
        'importance': mean_importance
    }).sort_values('importance', ascending=False).reset_index(drop=True)

    return importance_df


# =============================================================================
# FEATURE SELECTION FUNCTIONS
# =============================================================================

def select_features_single(top_k: int = 20):
    """
    Original function signature preserved for backward compatibility.
    Compute SHAP-based feature rankings for each modality.
    Saves top_k features to text files under '<project_root>/data/processed/selected_features/'.

    Args:
        top_k: Number of top features to select (default: 20)
    """
    select_features_multi(top_k_values=[top_k])


def select_features_multi(
        top_k_values: List[Union[int, str]] = [10, 20, 30, 40, 50, 60, 70, 80, 'all'],
        modalities: Optional[List[str]] = None,
        save_importance_csv: bool = True,
        verbose: bool = True
):
    """
    Compute SHAP-based feature rankings for multiple top_k values (ablation study).

    This function:
    1. Computes SHAP importance ONCE per modality (efficient)
    2. Saves feature lists for each top_k value
    3. Optionally saves full importance rankings to CSV

    Args:
        top_k_values: List of top_k values. Use 'all' for all features.
                      Default: [10, 20, 30, 40, 50, 60, 70, 80, 'all']
        modalities: List of modalities to process. None = all from config.
        save_importance_csv: Whether to save full SHAP importance rankings.
        verbose: Whether to print progress messages.

    Output files (original naming convention preserved):
        - <modality>_top<k>_features.txt (for each k in top_k_values)
        - <modality>_topall_features.txt (when 'all' is specified)
        - <modality>_shap_importance.csv (full rankings, if save_importance_csv=True)
    """
    # Load configuration
    cfg = load_config()

    feat_dir = cfg['features']['output_dir']
    if modalities is None:
        modalities = cfg['modalities']

    # Output directory (sibling of 'features')
    out_dir = Path(feat_dir).parent / 'selected_features'
    out_dir.mkdir(parents=True, exist_ok=True)

    # Also create SHAP_Features directory for importance CSVs
    shap_dir = Path(feat_dir) / 'SHAP_Features'
    shap_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("=" * 70)
        print("SHAP FEATURE SELECTION")
        print("=" * 70)
        print(f"Top-K values: {top_k_values}")
        print(f"Modalities: {modalities}")
        print(f"Output directory: {out_dir}")
        print("=" * 70)

    # Process each modality
    for mod in modalities:
        feat_file = Path(feat_dir) / f"{mod}_features.csv"

        if not feat_file.exists():
            if verbose:
                print(f"\n[WARNING] Missing features for '{mod}', skipping.")
            continue

        if verbose:
            print(f"\n[{mod}] Processing...")

        # Load features
        df = pd.read_csv(feat_file)
        X = df.drop(columns=['segment_id', 'label'], errors='ignore')
        y = df['label']

        total_features = len(X.columns)
        if verbose:
            print(f"  Total features available: {total_features}")

        # Compute SHAP importance (ONCE per modality - efficient!)
        if verbose:
            print(f"  Computing SHAP importance...")

        try:
            importance_df = compute_shap_importance(X, y)
        except Exception as e:
            if verbose:
                print(f"  [ERROR] SHAP computation failed: {e}")
            continue

        # Save full importance CSV (optional)
        if save_importance_csv:
            importance_csv = shap_dir / f"{mod}_shap_importance.csv"
            importance_df.to_csv(importance_csv, index=False)
            if verbose:
                print(f"  Saved importance rankings → {importance_csv}")

        # Save feature lists for each top_k value
        for top_k in top_k_values:
            if top_k == 'all':
                # Use all features
                k = total_features
                suffix = 'all'
                top_feats = importance_df['feature'].tolist()
            else:
                # Use top_k features
                k = min(int(top_k), total_features)
                suffix = str(top_k)
                top_feats = importance_df['feature'].head(k).tolist()

            # Write feature list (original naming convention)
            out_file = out_dir / f"{mod}_top{suffix}_features.txt"
            with open(out_file, 'w', encoding='utf-8') as wf:
                wf.write("\n".join(top_feats))

            if verbose:
                print(f"  Saved top {suffix} features ({len(top_feats)}) → {out_file}")

    if verbose:
        print("\n" + "=" * 70)
        print("FEATURE SELECTION COMPLETE")
        print("=" * 70)


def select_features(top_k: Union[int, List[Union[int, str]]] = 20):
    """
    Main entry point for feature selection (backward compatible).

    Args:
        top_k: Either a single integer (original behavior) or a list of values
               for ablation study. Use 'all' in list for all features.

    Examples:
        # Original behavior (single top_k)
        select_features(top_k=20)

        # Ablation study (multiple top_k values)
        select_features(top_k=[10, 20, 30, 40, 50, 60, 70, 80, 'all'])
    """
    if isinstance(top_k, list):
        select_features_multi(top_k_values=top_k)
    else:
        select_features_multi(top_k_values=[top_k])


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_selected_features(modality: str, top_k: Union[int, str] = 20, cfg: Optional[dict] = None) -> List[str]:
    """
    Load selected features for a given modality and top_k value.

    Args:
        modality: Modality name (e.g., 'EEG', 'ECG_EEG_Pupil')
        top_k: Number of features or 'all'
        cfg: Optional config dict (loads if not provided)

    Returns:
        List of feature names
    """
    if cfg is None:
        cfg = load_config()

    feat_dir = cfg['features']['output_dir']
    sel_dir = Path(feat_dir).parent / 'selected_features'

    suffix = 'all' if top_k == 'all' else str(top_k)
    sel_file = sel_dir / f"{modality}_top{suffix}_features.txt"

    if not sel_file.exists():
        raise FileNotFoundError(f"Selected features file not found: {sel_file}")

    with open(sel_file, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


def get_available_top_k_values(modality: str, cfg: Optional[dict] = None) -> List[Union[int, str]]:
    """
    Get list of available top_k values for a modality.

    Args:
        modality: Modality name
        cfg: Optional config dict

    Returns:
        List of available top_k values (integers and 'all')
    """
    if cfg is None:
        cfg = load_config()

    feat_dir = cfg['features']['output_dir']
    sel_dir = Path(feat_dir).parent / 'selected_features'

    available = []
    for f in sel_dir.glob(f"{modality}_top*_features.txt"):
        # Extract top_k value from filename
        name = f.stem  # e.g., "EEG_top20_features"
        parts = name.split('_top')
        if len(parts) == 2:
            k_str = parts[1].replace('_features', '')
            if k_str == 'all':
                available.append('all')
            else:
                try:
                    available.append(int(k_str))
                except ValueError:
                    pass

    # Sort: integers first (ascending), then 'all'
    int_vals = sorted([v for v in available if isinstance(v, int)])
    str_vals = [v for v in available if isinstance(v, str)]
    return int_vals + str_vals


def print_feature_selection_summary(cfg: Optional[dict] = None):
    """
    Print summary of all available selected feature files.
    """
    if cfg is None:
        cfg = load_config()

    feat_dir = cfg['features']['output_dir']
    sel_dir = Path(feat_dir).parent / 'selected_features'

    print("=" * 70)
    print("FEATURE SELECTION SUMMARY")
    print("=" * 70)
    print(f"Directory: {sel_dir}")
    print("-" * 70)

    for mod in cfg['modalities']:
        top_k_vals = get_available_top_k_values(mod, cfg)
        if top_k_vals:
            print(f"\n{mod}:")
            for k in top_k_vals:
                try:
                    feats = get_selected_features(mod, k, cfg)
                    print(f"  top_{k}: {len(feats)} features")
                except:
                    pass

    print("\n" + "=" * 70)


# =============================================================================
# ABLATION STUDY HELPER
# =============================================================================

def run_ablation_feature_selection(
        top_k_values: List[Union[int, str]] = [10, 20, 30, 40, 50, 60, 70, 80, 'all']
):
    """
    Convenience function to run feature selection for ablation study.

    This runs SHAP feature selection for multiple top_k values across all modalities.

    Args:
        top_k_values: List of top_k values to generate.
                      Default: [10, 20, 30, 40, 50, 60, 70, 80, 'all']
    """
    print("\n" + "=" * 70)
    print("ABLATION STUDY: FEATURE SELECTION")
    print("=" * 70)
    print(f"Generating feature lists for top_k = {top_k_values}")
    print("=" * 70 + "\n")

    select_features_multi(
        top_k_values=top_k_values,
        save_importance_csv=True,
        verbose=True
    )

    print("\n✅ Ablation feature selection complete!")
    print("   You can now run training and evaluation for each top_k value.")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='SHAP-based Feature Selection for Attention Detection System'
    )
    parser.add_argument(
        '--top_k',
        type=str,
        default='20',
        help='Top-K value(s). Use comma-separated for multiple (e.g., "10,20,30,all"). Default: 20'
    )
    parser.add_argument(
        '--ablation',
        action='store_true',
        help='Run full ablation study with default top_k values [10,20,30,40,50,60,70,80,all]'
    )

    args = parser.parse_args()

    if args.ablation:
        # Run full ablation study
        run_ablation_feature_selection()
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
            top_k_values = [20]  # Default

        if len(top_k_values) == 1:
            # Single value (original behavior)
            select_features(top_k=top_k_values[0])
        else:
            # Multiple values (ablation)
            select_features(top_k=top_k_values)