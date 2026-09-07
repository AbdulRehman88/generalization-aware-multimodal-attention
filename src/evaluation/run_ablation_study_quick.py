# =============================================================================
# run_ablation_study_quick.py
# =============================================================================
# Quick Ablation Study - LOSO and Stratified K-Fold ONLY
#
# This is a streamlined version for fast results:
#   - LOSO (Leave-One-Subject-Out) Cross-Validation
#   - Stratified K-Fold Cross-Validation (5-fold)
#
# Output saved to: outputs/summary/ablation_study_quick/
# (Separate from main ablation study to avoid overwrite)
#
# Author: Sanjar / Dr. Abdul Rehman Lab
# Date: November 2025
# Project: ETRI Attention Detection System
# =============================================================================

import os
import sys
import time
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Union, List, Optional, Dict, Tuple
from datetime import datetime

# Sklearn imports
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
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

# Output directory name (separate from main ablation study)
OUTPUT_DIR_NAME = "ablation_study_quick"


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
# DATA LOADING
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
    """Load features, labels, and subject IDs for a modality."""
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
# EVALUATION METHODS (LOSO + K-Fold ONLY)
# =============================================================================

def evaluate_loso(
        X: pd.DataFrame,
        y: pd.Series,
        subjects: pd.Series,
        clf_name: str
) -> Dict[str, float]:
    """Leave-One-Subject-Out Cross-Validation."""
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


# =============================================================================
# MAIN QUICK ABLATION STUDY
# =============================================================================

def run_quick_ablation_study(
        top_k_values: List[Union[int, str]] = [10, 20, 30, 40, 50, 60, 70, 80, 'all'],
        modalities: Optional[List[str]] = None,
        classifiers: List[str] = ['xgb', 'rf', 'et', 'lgbm', 'cat'],
        verbose: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    Run QUICK ablation study (LOSO + Stratified K-Fold ONLY).

    Args:
        top_k_values: List of top_k values to evaluate
        modalities: List of modalities (None = all from config)
        classifiers: List of classifier names
        verbose: Print progress

    Returns:
        Dictionary with 'loso' and 'stratified_kfold' DataFrames
    """
    cfg = load_config()

    if modalities is None:
        modalities = cfg['modalities']

    # Create SEPARATE output directory
    ablation_dir = os.path.join(cfg['summary']['output_dir'], OUTPUT_DIR_NAME)
    os.makedirs(ablation_dir, exist_ok=True)

    if verbose:
        print("=" * 80)
        print("QUICK ABLATION STUDY (LOSO + K-Fold Only)")
        print("=" * 80)
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Top-K values: {top_k_values}")
        print(f"Modalities: {modalities}")
        print(f"Classifiers: {classifiers}")
        print(f"Output directory: {ablation_dir}")
        print("=" * 80)

    # Initialize result containers
    results_loso = []
    results_kfold = []

    total_configs = len(top_k_values) * len(modalities) * len(classifiers)
    config_count = 0
    start_time = time.time()

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
                elapsed = time.time() - start_time
                avg_time = elapsed / config_count if config_count > 0 else 0
                remaining = avg_time * (total_configs - config_count)

                if verbose:
                    print(f"  [{config_count}/{total_configs}] {clf_name.upper()} "
                          f"(~{remaining / 60:.1f} min left)...", end=" ")

                base_record = {
                    'modality': mod,
                    'classifier': clf_name,
                    'top_k': suffix,
                    'n_features': n_features
                }

                # LOSO Evaluation
                try:
                    loso_metrics = evaluate_loso(X, y, subjects, clf_name)
                    results_loso.append({**base_record, **loso_metrics})
                    loso_acc = loso_metrics['accuracy_mean']
                except Exception as e:
                    loso_acc = 0
                    if verbose:
                        print(f"[LOSO ERR]", end=" ")

                # Stratified K-Fold Evaluation
                try:
                    kfold_metrics = evaluate_stratified_kfold(X, y, clf_name)
                    results_kfold.append({**base_record, **kfold_metrics})
                    kfold_acc = kfold_metrics['accuracy_mean']
                except Exception as e:
                    kfold_acc = 0
                    if verbose:
                        print(f"[KFold ERR]", end=" ")

                if verbose:
                    print(f"LOSO={loso_acc:.4f}, KFold={kfold_acc:.4f}")

    # Convert to DataFrames
    df_loso = pd.DataFrame(results_loso)
    df_kfold = pd.DataFrame(results_kfold)

    # Save CSVs
    loso_path = os.path.join(ablation_dir, 'ablation_loso.csv')
    kfold_path = os.path.join(ablation_dir, 'ablation_stratified_kfold.csv')

    df_loso.to_csv(loso_path, index=False)
    df_kfold.to_csv(kfold_path, index=False)

    if verbose:
        print(f"\n✅ Saved: {loso_path}")
        print(f"✅ Saved: {kfold_path}")

    # Generate summary report
    generate_quick_summary(df_loso, df_kfold, ablation_dir, verbose)

    # Generate visualizations
    generate_quick_visualizations(df_loso, df_kfold, ablation_dir, verbose)

    # Generate LaTeX table
    generate_latex_tables(df_loso, df_kfold, ablation_dir, verbose)

    total_time = time.time() - start_time

    if verbose:
        print("\n" + "=" * 80)
        print("QUICK ABLATION STUDY COMPLETE")
        print("=" * 80)
        print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total time: {total_time / 60:.1f} minutes")
        print(f"Results saved to: {ablation_dir}")
        print("=" * 80)

    return {'loso': df_loso, 'stratified_kfold': df_kfold}


# =============================================================================
# SUMMARY REPORT
# =============================================================================

def generate_quick_summary(
        df_loso: pd.DataFrame,
        df_kfold: pd.DataFrame,
        output_dir: str,
        verbose: bool = True
):
    """Generate summary report."""

    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("QUICK ABLATION STUDY - SUMMARY REPORT")
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("=" * 80)

    for method_name, df in [('LOSO', df_loso), ('Stratified K-Fold', df_kfold)]:
        if df.empty:
            continue

        report_lines.append(f"\n{'=' * 60}")
        report_lines.append(f"{method_name} RESULTS")
        report_lines.append(f"{'=' * 60}")

        # Overall best
        best_idx = df['accuracy_mean'].idxmax()
        best = df.loc[best_idx]
        report_lines.append(f"\n🏆 OVERALL BEST:")
        report_lines.append(f"   Modality: {best['modality']}")
        report_lines.append(f"   Classifier: {best['classifier'].upper()}")
        report_lines.append(f"   Top-K: {best['top_k']}")
        report_lines.append(f"   Accuracy: {best['accuracy_mean']:.4f} ± {best['accuracy_std']:.4f}")
        report_lines.append(f"   F1-Score: {best['f1_mean']:.4f}")

        # Best per modality
        report_lines.append(f"\n📊 BEST PER MODALITY:")
        for mod in sorted(df['modality'].unique()):
            mod_df = df[df['modality'] == mod]
            best_mod_idx = mod_df['accuracy_mean'].idxmax()
            best_mod = mod_df.loc[best_mod_idx]
            report_lines.append(
                f"   {mod}: {best_mod['classifier'].upper()} (top_{best_mod['top_k']}) "
                f"= {best_mod['accuracy_mean']:.4f}"
            )

        # Best top_k
        report_lines.append(f"\n📈 AVERAGE ACCURACY BY TOP-K:")
        for top_k in sorted(df['top_k'].unique(), key=lambda x: (x == 'all', int(x) if x != 'all' else 999)):
            topk_df = df[df['top_k'] == top_k]
            avg_acc = topk_df['accuracy_mean'].mean()
            best_acc = topk_df['accuracy_mean'].max()
            report_lines.append(f"   Top-{top_k}: avg={avg_acc:.4f}, best={best_acc:.4f}")

    # LOSO vs K-Fold comparison
    if not df_loso.empty and not df_kfold.empty:
        report_lines.append(f"\n{'=' * 60}")
        report_lines.append("LOSO vs K-Fold COMPARISON (Generalization Gap)")
        report_lines.append(f"{'=' * 60}")

        for mod in sorted(df_loso['modality'].unique()):
            loso_best = df_loso[df_loso['modality'] == mod]['accuracy_mean'].max()
            kfold_best = df_kfold[df_kfold['modality'] == mod]['accuracy_mean'].max()
            gap = kfold_best - loso_best
            report_lines.append(f"   {mod}: K-Fold={kfold_best:.4f}, LOSO={loso_best:.4f}, Gap={gap:.4f}")

    # Save report
    report_text = "\n".join(report_lines)
    report_path = os.path.join(output_dir, 'ablation_summary_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)

    if verbose:
        print("\n" + report_text)
        print(f"\n✅ Report saved: {report_path}")


# =============================================================================
# LATEX TABLES
# =============================================================================

def generate_latex_tables(
        df_loso: pd.DataFrame,
        df_kfold: pd.DataFrame,
        output_dir: str,
        verbose: bool = True
):
    """Generate LaTeX tables for publication."""

    for method_name, df in [('kfold', df_kfold), ('loso', df_loso)]:
        if df.empty:
            continue

        # Create pivot: best accuracy per modality × top_k
        pivot = df.pivot_table(
            index='modality',
            columns='top_k',
            values='accuracy_mean',
            aggfunc='max'
        )

        # Sort columns
        col_order = sorted(pivot.columns, key=lambda x: (x == 'all', int(x) if x != 'all' else 999))
        pivot = pivot[col_order]

        # Generate LaTeX
        latex_lines = []
        latex_lines.append(f"% LaTeX Table: {method_name.upper()} Ablation Results")
        latex_lines.append("\\begin{table}[htbp]")
        latex_lines.append("\\centering")
        latex_lines.append(f"\\caption{{Classification Accuracy (\\%) - {method_name.upper()}}}")
        latex_lines.append(f"\\label{{tab:ablation_{method_name}}}")
        latex_lines.append("\\begin{tabular}{l" + "c" * len(col_order) + "}")
        latex_lines.append("\\toprule")

        # Header
        header = "Modality & " + " & ".join([f"Top-{k}" for k in col_order]) + " \\\\"
        latex_lines.append(header)
        latex_lines.append("\\midrule")

        # Find global best for highlighting
        global_best = pivot.max().max()

        # Data rows
        for mod in pivot.index:
            row_vals = [mod.replace('_', '+')]
            for col in col_order:
                val = pivot.loc[mod, col]
                if pd.isna(val):
                    row_vals.append("-")
                elif val == global_best:
                    row_vals.append(f"\\textbf{{{val * 100:.1f}}}")
                else:
                    row_vals.append(f"{val * 100:.1f}")
            latex_lines.append(" & ".join(row_vals) + " \\\\")

        latex_lines.append("\\bottomrule")
        latex_lines.append("\\end{tabular}")
        latex_lines.append("\\end{table}")

        # Save
        latex_path = os.path.join(output_dir, f'table_{method_name}.tex')
        with open(latex_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(latex_lines))

        if verbose:
            print(f"✅ LaTeX table saved: {latex_path}")


# =============================================================================
# VISUALIZATIONS
# =============================================================================

def generate_quick_visualizations(
        df_loso: pd.DataFrame,
        df_kfold: pd.DataFrame,
        output_dir: str,
        verbose: bool = True
):
    """Generate publication-ready visualizations."""

    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 12

    fig_dir = os.path.join(output_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)

    # 1. Accuracy vs Top-K (Line Plot)
    if not df_kfold.empty:
        plt.figure(figsize=(12, 6))

        for mod in sorted(df_kfold['modality'].unique()):
            mod_df = df_kfold[df_kfold['modality'] == mod]
            best_per_topk = mod_df.groupby('top_k')['accuracy_mean'].max().reset_index()
            best_per_topk['sort_key'] = best_per_topk['top_k'].apply(
                lambda x: 999 if x == 'all' else int(x)
            )
            best_per_topk = best_per_topk.sort_values('sort_key')

            plt.plot(
                range(len(best_per_topk)),
                best_per_topk['accuracy_mean'] * 100,
                marker='o',
                label=mod.replace('_', '+'),
                linewidth=2,
                markersize=8
            )

        top_k_labels = sorted(df_kfold['top_k'].unique(),
                              key=lambda x: (x == 'all', int(x) if x != 'all' else 999))
        plt.xticks(range(len(top_k_labels)), [f"Top-{k}" for k in top_k_labels], fontweight='bold')
        plt.xlabel('Number of Features', fontweight='bold', fontsize=14)
        plt.ylabel('Accuracy (%)', fontweight='bold', fontsize=14)
        plt.title('Classification Accuracy vs Feature Count (Stratified K-Fold)', fontweight='bold', fontsize=14)
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        fig_path = os.path.join(fig_dir, 'accuracy_vs_topk_kfold.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()

        if verbose:
            print(f"✅ Saved: {fig_path}")

    # 2. Heatmap: K-Fold
    if not df_kfold.empty:
        pivot = df_kfold.pivot_table(
            index='modality',
            columns='top_k',
            values='accuracy_mean',
            aggfunc='max'
        )
        col_order = sorted(pivot.columns, key=lambda x: (x == 'all', int(x) if x != 'all' else 999))
        pivot = pivot[col_order] * 100

        plt.figure(figsize=(14, 8))
        sns.heatmap(
            pivot,
            annot=True,
            fmt='.1f',
            cmap='RdYlGn',
            center=pivot.values.mean(),
            linewidths=0.5,
            cbar_kws={'label': 'Accuracy (%)'},
            annot_kws={'fontsize': 11, 'fontweight': 'bold'}
        )
        plt.title('K-Fold Accuracy Heatmap (%)', fontweight='bold', fontsize=14)
        plt.xlabel('Number of Features (Top-K)', fontweight='bold', fontsize=12)
        plt.ylabel('Modality', fontweight='bold', fontsize=12)
        plt.tight_layout()

        fig_path = os.path.join(fig_dir, 'heatmap_kfold.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()

        if verbose:
            print(f"✅ Saved: {fig_path}")

    # 3. Heatmap: LOSO
    if not df_loso.empty:
        pivot = df_loso.pivot_table(
            index='modality',
            columns='top_k',
            values='accuracy_mean',
            aggfunc='max'
        )
        col_order = sorted(pivot.columns, key=lambda x: (x == 'all', int(x) if x != 'all' else 999))
        pivot = pivot[col_order] * 100

        plt.figure(figsize=(14, 8))
        sns.heatmap(
            pivot,
            annot=True,
            fmt='.1f',
            cmap='RdYlGn',
            center=pivot.values.mean(),
            linewidths=0.5,
            cbar_kws={'label': 'Accuracy (%)'},
            annot_kws={'fontsize': 11, 'fontweight': 'bold'}
        )
        plt.title('LOSO Accuracy Heatmap (%)', fontweight='bold', fontsize=14)
        plt.xlabel('Number of Features (Top-K)', fontweight='bold', fontsize=12)
        plt.ylabel('Modality', fontweight='bold', fontsize=12)
        plt.tight_layout()

        fig_path = os.path.join(fig_dir, 'heatmap_loso.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()

        if verbose:
            print(f"✅ Saved: {fig_path}")

    # 4. LOSO vs K-Fold Comparison Bar Chart
    if not df_loso.empty and not df_kfold.empty:
        modalities = sorted(df_loso['modality'].unique())

        loso_best = [df_loso[df_loso['modality'] == m]['accuracy_mean'].max() * 100 for m in modalities]
        kfold_best = [df_kfold[df_kfold['modality'] == m]['accuracy_mean'].max() * 100 for m in modalities]

        x = np.arange(len(modalities))
        width = 0.35

        fig, ax = plt.subplots(figsize=(14, 6))
        bars1 = ax.bar(x - width / 2, kfold_best, width, label='Stratified K-Fold', color='steelblue')
        bars2 = ax.bar(x + width / 2, loso_best, width, label='LOSO', color='coral')

        ax.set_xlabel('Modality', fontweight='bold', fontsize=12)
        ax.set_ylabel('Accuracy (%)', fontweight='bold', fontsize=12)
        ax.set_title('K-Fold vs LOSO: Best Accuracy per Modality', fontweight='bold', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace('_', '+') for m in modalities], rotation=45, ha='right', fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')
        for bar in bars2:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

        plt.tight_layout()

        fig_path = os.path.join(fig_dir, 'kfold_vs_loso_comparison.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()

        if verbose:
            print(f"✅ Saved: {fig_path}")

    # 5. Best Classifier per Modality (K-Fold)
    if not df_kfold.empty:
        # Find best classifier for each modality
        best_clf = df_kfold.loc[df_kfold.groupby('modality')['accuracy_mean'].idxmax()]

        fig, ax = plt.subplots(figsize=(12, 6))
        colors = plt.cm.Set2(np.linspace(0, 1, len(best_clf)))

        bars = ax.bar(
            [m.replace('_', '+') for m in best_clf['modality']],
            best_clf['accuracy_mean'] * 100,
            color=colors,
            edgecolor='black'
        )

        # Add classifier labels on bars
        for bar, clf in zip(bars, best_clf['classifier']):
            height = bar.get_height()
            ax.annotate(f'{clf.upper()}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=10, fontweight='bold')

        ax.set_xlabel('Modality', fontweight='bold', fontsize=12)
        ax.set_ylabel('Accuracy (%)', fontweight='bold', fontsize=12)
        ax.set_title('Best Classifier per Modality (K-Fold)', fontweight='bold', fontsize=14)
        plt.xticks(rotation=45, ha='right', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()

        fig_path = os.path.join(fig_dir, 'best_classifier_per_modality.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()

        if verbose:
            print(f"✅ Saved: {fig_path}")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    print("\n" + "🚀" * 30)
    print("STARTING QUICK ABLATION STUDY")
    print("🚀" * 30 + "\n")

    results = run_quick_ablation_study(
        top_k_values=[10, 20, 30, 40, 50, 60, 70, 80, 'all'],
        classifiers=['xgb', 'rf', 'et', 'lgbm', 'cat'],
        verbose=True
    )

    print("\n" + "✅" * 30)
    print("QUICK ABLATION STUDY FINISHED!")
    print("✅" * 30)