# =============================================================================
# run_ablation_study.py
# =============================================================================
# Unified Ablation Study Evaluation Pipeline for Attention Detection System
#
# This script performs comprehensive evaluation across all top_k configurations:
#   - LOSO (Leave-One-Subject-Out) Cross-Validation
#   - Stratified K-Fold Cross-Validation
#   - Repeated Stratified K-Fold
#   - Bootstrap Confidence Intervals
#   - End-to-End Latency Benchmarking
#
# Outputs:
#   - Detailed metrics CSV for each evaluation method
#   - Publication-ready comparison tables (LaTeX format)
#   - Visualization figures (bar plots, heatmaps)
#   - Summary report with best configurations
#
# Author: Sanjar / Dr. Abdul Rehman Lab
# Date: November 2025
# Project: ETRI Attention Detection System
# =============================================================================

import os
import sys
import time
import yaml
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Union, List, Optional, Dict, Tuple
from datetime import datetime

# Sklearn imports
from sklearn.model_selection import (
    StratifiedKFold,
    RepeatedStratifiedKFold,
    LeaveOneGroupOut
)
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# Model imports
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

import warnings

warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURATION
# =============================================================================

def load_config(cfg_path: Optional[str] = None) -> dict:
    """Load configuration from config.yaml"""
    if cfg_path is None:
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
# DATA LOADING UTILITIES
# =============================================================================

def extract_subject_id(segment_id: str) -> str:
    """Extract subject ID from segment filename: '1.P01_seg0.csv' -> 'P01'"""
    try:
        part = segment_id.split(".")[1]
        return part.split("_")[0]
    except:
        return "unknown"


def load_features_and_labels(
        modality: str,
        top_k: Union[int, str],
        cfg: dict
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Load features, labels, and subject IDs for a modality.

    Returns:
        X: Feature DataFrame
        y: Labels Series
        subjects: Subject IDs Series
    """
    feat_dir = cfg['features']['output_dir']
    feat_file = os.path.join(feat_dir, f"{modality}_features.csv")

    if not os.path.exists(feat_file):
        raise FileNotFoundError(f"Feature file not found: {feat_file}")

    df = pd.read_csv(feat_file)

    # Load selected features
    suffix = 'all' if top_k == 'all' else str(top_k)
    sel_path = os.path.join(
        os.path.dirname(feat_dir),
        'selected_features',
        f"{modality}_top{suffix}_features.txt"
    )

    with open(sel_path, encoding='utf-8') as f:
        sel_feats = [line.strip() for line in f if line.strip()]

    # Filter to available features
    sel_feats = [f for f in sel_feats if f in df.columns]

    X = df[sel_feats]
    y = df['label']

    # Extract subject IDs
    if 'segment_id' in df.columns:
        subjects = df['segment_id'].apply(extract_subject_id)
    elif 'Segment_ID' in df.columns:
        subjects = df['Segment_ID'].apply(extract_subject_id)
    else:
        subjects = pd.Series(['unknown'] * len(df))

    return X, y, subjects


def get_classifier(name: str, random_state: int = 42):
    """Get a fresh classifier instance by name."""
    classifiers = {
        'xgb': XGBClassifier(
            use_label_encoder=False,
            eval_metric='mlogloss',
            random_state=random_state,
            n_estimators=100,
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
    return classifiers.get(name.lower())


# =============================================================================
# EVALUATION METHODS
# =============================================================================

def evaluate_loso(
        X: pd.DataFrame,
        y: pd.Series,
        subjects: pd.Series,
        clf_name: str
) -> Dict[str, float]:
    """
    Leave-One-Subject-Out Cross-Validation.

    Returns:
        Dictionary with mean metrics and std
    """
    logo = LeaveOneGroupOut()

    accs, precs, recs, f1s, aucs = [], [], [], [], []

    for train_idx, test_idx in logo.split(X, y, subjects):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Scale features
        scaler = StandardScaler()
        X_train_s = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        X_test_s = pd.DataFrame(
            scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )

        # Train and predict
        clf = get_classifier(clf_name)
        clf.fit(X_train_s, y_train)
        y_pred = clf.predict(X_test_s)
        y_proba = clf.predict_proba(X_test_s)

        # Metrics
        accs.append(accuracy_score(y_test, y_pred))
        precs.append(precision_score(y_test, y_pred, average='macro', zero_division=0))
        recs.append(recall_score(y_test, y_pred, average='macro', zero_division=0))
        f1s.append(f1_score(y_test, y_pred, average='macro', zero_division=0))

        # AUC
        try:
            classes = np.unique(y)
            if len(classes) > 2:
                y_test_bin = label_binarize(y_test, classes=classes)
                auc = roc_auc_score(y_test_bin, y_proba, average='macro', multi_class='ovr')
            else:
                auc = roc_auc_score(y_test, y_proba[:, 1])
            aucs.append(auc)
        except:
            pass

    return {
        'accuracy_mean': np.mean(accs),
        'accuracy_std': np.std(accs),
        'precision_mean': np.mean(precs),
        'precision_std': np.std(precs),
        'recall_mean': np.mean(recs),
        'recall_std': np.std(recs),
        'f1_mean': np.mean(f1s),
        'f1_std': np.std(f1s),
        'roc_auc_mean': np.mean(aucs) if aucs else np.nan,
        'roc_auc_std': np.std(aucs) if aucs else np.nan
    }


def evaluate_stratified_kfold(
        X: pd.DataFrame,
        y: pd.Series,
        clf_name: str,
        n_splits: int = 5
) -> Dict[str, float]:
    """Stratified K-Fold Cross-Validation."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    accs, precs, recs, f1s, aucs = [], [], [], [], []

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        clf = get_classifier(clf_name)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)

        accs.append(accuracy_score(y_test, y_pred))
        precs.append(precision_score(y_test, y_pred, average='macro', zero_division=0))
        recs.append(recall_score(y_test, y_pred, average='macro', zero_division=0))
        f1s.append(f1_score(y_test, y_pred, average='macro', zero_division=0))

        try:
            classes = np.unique(y)
            if len(classes) > 2:
                y_test_bin = label_binarize(y_test, classes=classes)
                auc = roc_auc_score(y_test_bin, y_proba, average='macro', multi_class='ovr')
            else:
                auc = roc_auc_score(y_test, y_proba[:, 1])
            aucs.append(auc)
        except:
            pass

    return {
        'accuracy_mean': np.mean(accs),
        'accuracy_std': np.std(accs),
        'precision_mean': np.mean(precs),
        'precision_std': np.std(precs),
        'recall_mean': np.mean(recs),
        'recall_std': np.std(recs),
        'f1_mean': np.mean(f1s),
        'f1_std': np.std(f1s),
        'roc_auc_mean': np.mean(aucs) if aucs else np.nan,
        'roc_auc_std': np.std(aucs) if aucs else np.nan
    }


def evaluate_repeated_kfold(
        X: pd.DataFrame,
        y: pd.Series,
        clf_name: str,
        n_splits: int = 5,
        n_repeats: int = 10
) -> Dict[str, float]:
    """Repeated Stratified K-Fold Cross-Validation."""
    rskf = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=42
    )

    accs, precs, recs, f1s = [], [], [], []

    for train_idx, test_idx in rskf.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        clf = get_classifier(clf_name)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        accs.append(accuracy_score(y_test, y_pred))
        precs.append(precision_score(y_test, y_pred, average='macro', zero_division=0))
        recs.append(recall_score(y_test, y_pred, average='macro', zero_division=0))
        f1s.append(f1_score(y_test, y_pred, average='macro', zero_division=0))

    return {
        'accuracy_mean': np.mean(accs),
        'accuracy_std': np.std(accs),
        'precision_mean': np.mean(precs),
        'precision_std': np.std(precs),
        'recall_mean': np.mean(recs),
        'recall_std': np.std(recs),
        'f1_mean': np.mean(f1s),
        'f1_std': np.std(f1s)
    }


def evaluate_bootstrap_ci(
        X: pd.DataFrame,
        y: pd.Series,
        clf_name: str,
        n_bootstrap: int = 1000
) -> Dict[str, float]:
    """Bootstrap Confidence Intervals."""
    rng = np.random.RandomState(42)
    accs = []

    for _ in range(n_bootstrap):
        idx = rng.choice(len(y), len(y), replace=True)
        X_boot, y_boot = X.iloc[idx], y.iloc[idx]

        clf = get_classifier(clf_name)
        clf.fit(X_boot, y_boot)
        y_pred = clf.predict(X)
        accs.append(accuracy_score(y, y_pred))

    return {
        'accuracy_mean': np.mean(accs),
        'accuracy_std': np.std(accs),
        'accuracy_ci_lower': np.percentile(accs, 2.5),
        'accuracy_ci_upper': np.percentile(accs, 97.5)
    }


def benchmark_latency(
        X: pd.DataFrame,
        clf_name: str,
        top_k: Union[int, str],
        cfg: dict,
        n_repeat: int = 100
) -> Dict[str, float]:
    """Benchmark inference latency."""
    # Load pre-trained model if available
    suffix = 'all' if top_k == 'all' else str(top_k)
    model_dir = os.path.join(cfg['models']['output_dir'], f"top_{suffix}")

    # Get a fresh classifier and fit on data (or load from disk)
    clf = get_classifier(clf_name)
    y_dummy = np.zeros(len(X))  # Just for timing
    clf.fit(X, y_dummy)

    # Single sample for inference
    X_single = X.iloc[[0]]

    # Warm-up
    for _ in range(10):
        _ = clf.predict(X_single)

    # Timed runs
    latencies = []
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        _ = clf.predict(X_single)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)  # ms

    return {
        'latency_mean_ms': np.mean(latencies),
        'latency_std_ms': np.std(latencies),
        'latency_min_ms': np.min(latencies),
        'latency_max_ms': np.max(latencies)
    }


# =============================================================================
# MAIN ABLATION STUDY
# =============================================================================

def run_ablation_study(
        top_k_values: List[Union[int, str]] = [10, 20, 30, 40, 50, 60, 70, 80, 'all'],
        modalities: Optional[List[str]] = None,
        classifiers: List[str] = ['xgb', 'rf', 'et', 'lgbm', 'cat'],
        eval_methods: List[str] = ['loso', 'stratified_kfold', 'repeated_kfold', 'bootstrap_ci'],
        run_latency: bool = True,
        verbose: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    Run comprehensive ablation study.

    Args:
        top_k_values: List of top_k values to evaluate
        modalities: List of modalities (None = all from config)
        classifiers: List of classifier names
        eval_methods: List of evaluation methods
        run_latency: Whether to run latency benchmarking
        verbose: Print progress

    Returns:
        Dictionary of result DataFrames for each evaluation method
    """
    cfg = load_config()

    if modalities is None:
        modalities = cfg['modalities']

    # Create output directory
    ablation_dir = os.path.join(cfg['summary']['output_dir'], 'ablation_study')
    os.makedirs(ablation_dir, exist_ok=True)

    if verbose:
        print("=" * 80)
        print("COMPREHENSIVE ABLATION STUDY")
        print("=" * 80)
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Top-K values: {top_k_values}")
        print(f"Modalities: {modalities}")
        print(f"Classifiers: {classifiers}")
        print(f"Evaluation methods: {eval_methods}")
        print(f"Output directory: {ablation_dir}")
        print("=" * 80)

    # Initialize result containers
    results = {method: [] for method in eval_methods}
    if run_latency:
        results['latency'] = []

    total_configs = len(top_k_values) * len(modalities) * len(classifiers)
    config_count = 0

    # Main loop
    for top_k in top_k_values:
        suffix = 'all' if top_k == 'all' else str(top_k)

        for mod in modalities:
            if verbose:
                print(f"\n[{mod}] Loading features (top_{suffix})...")

            try:
                X, y, subjects = load_features_and_labels(mod, top_k, cfg)
                n_features = len(X.columns)

                if verbose:
                    print(f"  Features: {n_features}, Samples: {len(X)}")

            except Exception as e:
                if verbose:
                    print(f"  [ERROR] Could not load data: {e}")
                continue

            for clf_name in classifiers:
                config_count += 1

                if verbose:
                    print(f"  [{config_count}/{total_configs}] {clf_name.upper()}...", end=" ")

                base_record = {
                    'modality': mod,
                    'classifier': clf_name,
                    'top_k': suffix,
                    'n_features': n_features
                }

                # Run each evaluation method
                for method in eval_methods:
                    try:
                        if method == 'loso':
                            metrics = evaluate_loso(X, y, subjects, clf_name)
                        elif method == 'stratified_kfold':
                            metrics = evaluate_stratified_kfold(X, y, clf_name)
                        elif method == 'repeated_kfold':
                            metrics = evaluate_repeated_kfold(X, y, clf_name)
                        elif method == 'bootstrap_ci':
                            metrics = evaluate_bootstrap_ci(X, y, clf_name, n_bootstrap=500)
                        else:
                            continue

                        record = {**base_record, **metrics}
                        results[method].append(record)

                    except Exception as e:
                        if verbose:
                            print(f"[{method} ERROR: {e}]", end=" ")

                # Latency benchmark
                if run_latency:
                    try:
                        latency = benchmark_latency(X, clf_name, top_k, cfg, n_repeat=50)
                        record = {**base_record, **latency}
                        results['latency'].append(record)
                    except Exception as e:
                        if verbose:
                            print(f"[latency ERROR: {e}]", end=" ")

                if verbose:
                    # Print accuracy from stratified_kfold if available
                    if 'stratified_kfold' in eval_methods and results['stratified_kfold']:
                        last = results['stratified_kfold'][-1]
                        print(f"Acc={last.get('accuracy_mean', 0):.4f}")
                    else:
                        print("Done")

    # Convert to DataFrames and save
    result_dfs = {}

    for method, records in results.items():
        if records:
            df = pd.DataFrame(records)
            result_dfs[method] = df

            # Save CSV
            csv_path = os.path.join(ablation_dir, f'ablation_{method}.csv')
            df.to_csv(csv_path, index=False)

            if verbose:
                print(f"\nSaved: {csv_path}")

    # Generate summary report
    if verbose:
        print("\n" + "=" * 80)
        print("GENERATING SUMMARY REPORT")
        print("=" * 80)

    generate_summary_report(result_dfs, ablation_dir, verbose)

    # Generate visualizations
    if verbose:
        print("\n" + "=" * 80)
        print("GENERATING VISUALIZATIONS")
        print("=" * 80)

    generate_visualizations(result_dfs, ablation_dir, verbose)

    if verbose:
        print("\n" + "=" * 80)
        print("ABLATION STUDY COMPLETE")
        print("=" * 80)
        print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Results saved to: {ablation_dir}")

    return result_dfs


# =============================================================================
# SUMMARY REPORT GENERATION
# =============================================================================

def generate_summary_report(
        result_dfs: Dict[str, pd.DataFrame],
        output_dir: str,
        verbose: bool = True
):
    """Generate summary report with best configurations."""

    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("ABLATION STUDY SUMMARY REPORT")
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("=" * 80)

    # Best configurations for each method
    for method, df in result_dfs.items():
        if df.empty or 'accuracy_mean' not in df.columns:
            continue

        report_lines.append(f"\n{'=' * 40}")
        report_lines.append(f"BEST CONFIGURATIONS - {method.upper()}")
        report_lines.append(f"{'=' * 40}")

        # Overall best
        best_idx = df['accuracy_mean'].idxmax()
        best = df.loc[best_idx]
        report_lines.append(f"\nOverall Best:")
        report_lines.append(f"  Modality: {best['modality']}")
        report_lines.append(f"  Classifier: {best['classifier']}")
        report_lines.append(f"  Top-K: {best['top_k']}")
        report_lines.append(f"  Accuracy: {best['accuracy_mean']:.4f} ± {best.get('accuracy_std', 0):.4f}")

        # Best per modality
        report_lines.append(f"\nBest per Modality:")
        for mod in df['modality'].unique():
            mod_df = df[df['modality'] == mod]
            best_mod_idx = mod_df['accuracy_mean'].idxmax()
            best_mod = mod_df.loc[best_mod_idx]
            report_lines.append(
                f"  {mod}: {best_mod['classifier']} (top_{best_mod['top_k']}) "
                f"= {best_mod['accuracy_mean']:.4f}"
            )

        # Best top_k overall
        report_lines.append(f"\nAverage Accuracy by Top-K:")
        for top_k in sorted(df['top_k'].unique(), key=lambda x: (x != 'all', int(x) if x != 'all' else 999)):
            topk_df = df[df['top_k'] == top_k]
            avg_acc = topk_df['accuracy_mean'].mean()
            report_lines.append(f"  top_{top_k}: {avg_acc:.4f}")

    # Latency summary
    if 'latency' in result_dfs and not result_dfs['latency'].empty:
        df_lat = result_dfs['latency']
        report_lines.append(f"\n{'=' * 40}")
        report_lines.append("LATENCY SUMMARY")
        report_lines.append(f"{'=' * 40}")

        # Fastest configurations
        fastest_idx = df_lat['latency_mean_ms'].idxmin()
        fastest = df_lat.loc[fastest_idx]
        report_lines.append(f"\nFastest Configuration:")
        report_lines.append(f"  {fastest['modality']} + {fastest['classifier']} (top_{fastest['top_k']})")
        report_lines.append(f"  Latency: {fastest['latency_mean_ms']:.3f} ms")

        # Average latency by classifier
        report_lines.append(f"\nAverage Latency by Classifier:")
        for clf in df_lat['classifier'].unique():
            avg_lat = df_lat[df_lat['classifier'] == clf]['latency_mean_ms'].mean()
            report_lines.append(f"  {clf}: {avg_lat:.3f} ms")

    # Save report
    report_text = "\n".join(report_lines)
    report_path = os.path.join(output_dir, 'ablation_summary_report.txt')
    with open(report_path, 'w') as f:
        f.write(report_text)

    if verbose:
        print(report_text)
        print(f"\nReport saved: {report_path}")

    # Generate LaTeX table
    generate_latex_table(result_dfs, output_dir, verbose)


def generate_latex_table(
        result_dfs: Dict[str, pd.DataFrame],
        output_dir: str,
        verbose: bool = True
):
    """Generate LaTeX-formatted tables for publication."""

    if 'stratified_kfold' not in result_dfs:
        return

    df = result_dfs['stratified_kfold']

    # Pivot table: modality x top_k (best classifier)
    latex_lines = []
    latex_lines.append("% LaTeX Table: Ablation Study Results")
    latex_lines.append("% Best accuracy for each modality and top_k configuration")
    latex_lines.append("\\begin{table}[htbp]")
    latex_lines.append("\\centering")
    latex_lines.append("\\caption{Ablation Study: Classification Accuracy by Feature Count}")
    latex_lines.append("\\label{tab:ablation}")

    top_k_values = sorted(df['top_k'].unique(), key=lambda x: (x != 'all', int(x) if x != 'all' else 999))
    n_cols = len(top_k_values) + 1

    latex_lines.append("\\begin{tabular}{l" + "c" * len(top_k_values) + "}")
    latex_lines.append("\\toprule")
    header = "Modality & " + " & ".join([f"Top-{k}" for k in top_k_values]) + " \\\\"
    latex_lines.append(header)
    latex_lines.append("\\midrule")

    for mod in df['modality'].unique():
        row_vals = [mod.replace('_', '\\_')]
        for top_k in top_k_values:
            subset = df[(df['modality'] == mod) & (df['top_k'] == top_k)]
            if not subset.empty:
                best_acc = subset['accuracy_mean'].max()
                row_vals.append(f"{best_acc:.3f}")
            else:
                row_vals.append("-")
        latex_lines.append(" & ".join(row_vals) + " \\\\")

    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\end{table}")

    latex_text = "\n".join(latex_lines)
    latex_path = os.path.join(output_dir, 'ablation_table.tex')
    with open(latex_path, 'w') as f:
        f.write(latex_text)

    if verbose:
        print(f"LaTeX table saved: {latex_path}")


# =============================================================================
# VISUALIZATION GENERATION
# =============================================================================

def generate_visualizations(
        result_dfs: Dict[str, pd.DataFrame],
        output_dir: str,
        verbose: bool = True
):
    """Generate publication-ready visualizations."""

    # Set style
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 12

    fig_dir = os.path.join(output_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)

    # 1. Accuracy vs Top-K (line plot)
    if 'stratified_kfold' in result_dfs:
        df = result_dfs['stratified_kfold']

        plt.figure(figsize=(12, 6))

        for mod in df['modality'].unique():
            mod_df = df[df['modality'] == mod]
            # Group by top_k, take best classifier
            best_per_topk = mod_df.groupby('top_k')['accuracy_mean'].max().reset_index()

            # Sort top_k values
            best_per_topk['sort_key'] = best_per_topk['top_k'].apply(
                lambda x: 999 if x == 'all' else int(x)
            )
            best_per_topk = best_per_topk.sort_values('sort_key')

            plt.plot(
                range(len(best_per_topk)),
                best_per_topk['accuracy_mean'],
                marker='o',
                label=mod,
                linewidth=2,
                markersize=8
            )

        top_k_labels = sorted(df['top_k'].unique(), key=lambda x: (x != 'all', int(x) if x != 'all' else 999))
        plt.xticks(range(len(top_k_labels)), [f"Top-{k}" for k in top_k_labels])
        plt.xlabel('Number of Features', fontweight='bold')
        plt.ylabel('Accuracy', fontweight='bold')
        plt.title('Classification Accuracy vs Feature Count', fontweight='bold')
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        fig_path = os.path.join(fig_dir, 'accuracy_vs_topk.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()

        if verbose:
            print(f"Saved: {fig_path}")

    # 2. Heatmap: Modality x Top-K
    if 'stratified_kfold' in result_dfs:
        df = result_dfs['stratified_kfold']

        # Create pivot table
        pivot_data = df.pivot_table(
            index='modality',
            columns='top_k',
            values='accuracy_mean',
            aggfunc='max'
        )

        # Sort columns
        col_order = sorted(pivot_data.columns, key=lambda x: (x != 'all', int(x) if x != 'all' else 999))
        pivot_data = pivot_data[col_order]

        plt.figure(figsize=(12, 8))
        sns.heatmap(
            pivot_data,
            annot=True,
            fmt='.3f',
            cmap='RdYlGn',
            center=pivot_data.values.mean(),
            linewidths=0.5,
            cbar_kws={'label': 'Accuracy'}
        )
        plt.title('Classification Accuracy Heatmap', fontweight='bold', fontsize=14)
        plt.xlabel('Number of Features (Top-K)', fontweight='bold')
        plt.ylabel('Modality', fontweight='bold')
        plt.tight_layout()

        fig_path = os.path.join(fig_dir, 'accuracy_heatmap.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()

        if verbose:
            print(f"Saved: {fig_path}")

    # 3. LOSO vs Stratified K-Fold comparison
    if 'loso' in result_dfs and 'stratified_kfold' in result_dfs:
        df_loso = result_dfs['loso']
        df_skf = result_dfs['stratified_kfold']

        # Merge for comparison
        df_loso['method'] = 'LOSO'
        df_skf['method'] = 'Stratified K-Fold'

        df_compare = pd.concat([
            df_loso[['modality', 'top_k', 'accuracy_mean', 'method']],
            df_skf[['modality', 'top_k', 'accuracy_mean', 'method']]
        ])

        # Best per modality for each method
        best_compare = df_compare.groupby(['modality', 'method'])['accuracy_mean'].max().reset_index()

        plt.figure(figsize=(12, 6))
        x = np.arange(len(best_compare['modality'].unique()))
        width = 0.35

        modalities = sorted(best_compare['modality'].unique())
        loso_vals = [
            best_compare[(best_compare['modality'] == m) & (best_compare['method'] == 'LOSO')]['accuracy_mean'].values[
                0]
            if len(best_compare[(best_compare['modality'] == m) & (best_compare['method'] == 'LOSO')]) > 0 else 0
            for m in modalities]
        skf_vals = [best_compare[(best_compare['modality'] == m) & (best_compare['method'] == 'Stratified K-Fold')][
                        'accuracy_mean'].values[0]
                    if len(
            best_compare[(best_compare['modality'] == m) & (best_compare['method'] == 'Stratified K-Fold')]) > 0 else 0
                    for m in modalities]

        plt.bar(x - width / 2, loso_vals, width, label='LOSO', color='steelblue')
        plt.bar(x + width / 2, skf_vals, width, label='Stratified K-Fold', color='coral')

        plt.xlabel('Modality', fontweight='bold')
        plt.ylabel('Accuracy', fontweight='bold')
        plt.title('LOSO vs Stratified K-Fold Comparison', fontweight='bold')
        plt.xticks(x, modalities, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()

        fig_path = os.path.join(fig_dir, 'loso_vs_kfold.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()

        if verbose:
            print(f"Saved: {fig_path}")

    # 4. Latency comparison
    if 'latency' in result_dfs:
        df_lat = result_dfs['latency']

        plt.figure(figsize=(14, 6))

        # Group by classifier
        clf_lat = df_lat.groupby('classifier')['latency_mean_ms'].mean().sort_values()

        colors = plt.cm.Pastel1(np.linspace(0, 1, len(clf_lat)))
        bars = plt.bar(clf_lat.index, clf_lat.values, color=colors, edgecolor='black')

        plt.xlabel('Classifier', fontweight='bold')
        plt.ylabel('Latency (ms)', fontweight='bold')
        plt.title('Average Inference Latency by Classifier', fontweight='bold')

        # Add value labels
        for bar, val in zip(bars, clf_lat.values):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                     f'{val:.2f}', ha='center', va='bottom', fontsize=10)

        plt.tight_layout()

        fig_path = os.path.join(fig_dir, 'latency_by_classifier.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()

        if verbose:
            print(f"Saved: {fig_path}")


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Run comprehensive ablation study for Attention Detection System'
    )
    parser.add_argument(
        '--top_k',
        type=str,
        default='10,20,30,40,50,60,70,80,all',
        help='Comma-separated top_k values (default: 10,20,30,40,50,60,70,80,all)'
    )
    parser.add_argument(
        '--methods',
        type=str,
        default='loso,stratified_kfold,repeated_kfold,bootstrap_ci',
        help='Comma-separated evaluation methods'
    )
    parser.add_argument(
        '--no-latency',
        action='store_true',
        help='Skip latency benchmarking'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick mode: only LOSO and stratified_kfold, reduced bootstrap'
    )

    args = parser.parse_args()

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
                pass

    # Parse methods
    methods = [m.strip() for m in args.methods.split(',')]

    if args.quick:
        methods = ['loso', 'stratified_kfold']

    # Run ablation study
    run_ablation_study(
        top_k_values=top_k_values,
        eval_methods=methods,
        run_latency=not args.no_latency,
        verbose=True
    )