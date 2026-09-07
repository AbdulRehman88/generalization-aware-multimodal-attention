# =============================================================================
# benchmark_latency.py
# =============================================================================
# Standalone Latency Benchmark for Ablation Study
#
# Measures inference latency for all Top-K configurations to determine
# optimal trade-off between accuracy and speed for clinical applications.
#
# Output: outputs/summary/ablation_study_quick/latency_benchmark.csv
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
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Union, List, Optional, Dict
from datetime import datetime

# Model imports
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

import warnings

warnings.filterwarnings('ignore')


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
            n_jobs=1  # Single thread for fair latency comparison
        ),
        'et': ExtraTreesClassifier(
            n_estimators=100,
            random_state=random_state,
            n_jobs=1  # Single thread for fair latency comparison
        ),
        'lgbm': LGBMClassifier(
            n_estimators=100,
            random_state=random_state,
            verbose=-1,
            n_jobs=1  # Single thread for fair latency comparison
        ),
        'cat': CatBoostClassifier(
            iterations=100,
            random_state=random_state,
            verbose=0,
            thread_count=1  # Single thread for fair latency comparison
        )
    }
    return classifiers.get(name.lower())


def load_features(modality: str, top_k: Union[int, str], cfg: dict) -> pd.DataFrame:
    """Load features for a modality and top_k configuration."""
    feat_dir = cfg['features']['output_dir']
    feat_file = os.path.join(feat_dir, f"{modality}_features.csv")

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

    return X, y


def benchmark_single_config(
        X: pd.DataFrame,
        y: pd.Series,
        clf_name: str,
        n_warmup: int = 10,
        n_repeat: int = 100
) -> Dict[str, float]:
    """
    Benchmark inference latency for a single configuration.

    Args:
        X: Feature DataFrame
        y: Labels
        clf_name: Classifier name
        n_warmup: Number of warmup iterations
        n_repeat: Number of timed iterations

    Returns:
        Dictionary with latency statistics
    """
    # Train classifier
    clf = get_classifier(clf_name)
    clf.fit(X, y)

    # Single sample for inference (simulating real-time prediction)
    X_single = X.iloc[[0]]

    # Warmup runs (not timed)
    for _ in range(n_warmup):
        _ = clf.predict(X_single)

    # Timed runs
    latencies = []
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        _ = clf.predict(X_single)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)  # Convert to milliseconds

    return {
        'latency_mean_ms': np.mean(latencies),
        'latency_std_ms': np.std(latencies),
        'latency_min_ms': np.min(latencies),
        'latency_max_ms': np.max(latencies),
        'latency_median_ms': np.median(latencies),
        'latency_p95_ms': np.percentile(latencies, 95),
        'latency_p99_ms': np.percentile(latencies, 99)
    }


def run_latency_benchmark(
        top_k_values: List[Union[int, str]] = [10, 20, 30, 40, 50, 60, 70, 80, 'all'],
        modalities: Optional[List[str]] = None,
        classifiers: List[str] = ['xgb', 'rf', 'et', 'lgbm', 'cat'],
        n_repeat: int = 100,
        verbose: bool = True
) -> pd.DataFrame:
    """
    Run comprehensive latency benchmark.

    Args:
        top_k_values: List of top_k values to benchmark
        modalities: List of modalities (None = all)
        classifiers: List of classifier names
        n_repeat: Number of inference repetitions per config
        verbose: Print progress

    Returns:
        DataFrame with all latency results
    """
    cfg = load_config()

    if modalities is None:
        modalities = cfg['modalities']

    # Output directory
    output_dir = os.path.join(cfg['summary']['output_dir'], 'ablation_study_quick')
    os.makedirs(output_dir, exist_ok=True)

    if verbose:
        print("=" * 70)
        print("LATENCY BENCHMARK FOR ABLATION STUDY")
        print("=" * 70)
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Top-K values: {top_k_values}")
        print(f"Modalities: {modalities}")
        print(f"Classifiers: {classifiers}")
        print(f"Repetitions per config: {n_repeat}")
        print("=" * 70)

    results = []
    total_configs = len(top_k_values) * len(modalities) * len(classifiers)
    config_count = 0
    start_time = time.time()

    for top_k in top_k_values:
        suffix = 'all' if top_k == 'all' else str(top_k)

        for mod in modalities:
            if verbose:
                print(f"\n[{mod}] Top-{suffix}...")

            try:
                X, y = load_features(mod, top_k, cfg)
                n_features = len(X.columns)
            except Exception as e:
                if verbose:
                    print(f"  [ERROR] Could not load data: {e}")
                continue

            for clf_name in classifiers:
                config_count += 1

                if verbose:
                    print(f"  [{config_count}/{total_configs}] {clf_name.upper()}...", end=" ")

                try:
                    latency_stats = benchmark_single_config(X, y, clf_name, n_repeat=n_repeat)

                    record = {
                        'modality': mod,
                        'classifier': clf_name,
                        'top_k': suffix,
                        'n_features': n_features,
                        **latency_stats
                    }
                    results.append(record)

                    if verbose:
                        print(f"{latency_stats['latency_mean_ms']:.3f} ± {latency_stats['latency_std_ms']:.3f} ms")

                except Exception as e:
                    if verbose:
                        print(f"[ERROR: {e}]")

    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Save results
    csv_path = os.path.join(output_dir, 'latency_benchmark.csv')
    df.to_csv(csv_path, index=False)

    if verbose:
        print(f"\n✅ Saved: {csv_path}")

    # Generate summary
    generate_latency_summary(df, output_dir, verbose)

    # Generate visualizations
    generate_latency_plots(df, output_dir, verbose)

    total_time = time.time() - start_time

    if verbose:
        print("\n" + "=" * 70)
        print("LATENCY BENCHMARK COMPLETE")
        print("=" * 70)
        print(f"Total time: {total_time:.1f} seconds")
        print(f"Results saved to: {output_dir}")
        print("=" * 70)

    return df


def generate_latency_summary(df: pd.DataFrame, output_dir: str, verbose: bool = True):
    """Generate latency summary report."""

    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("LATENCY BENCHMARK SUMMARY")
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("=" * 70)

    # Overall fastest
    fastest_idx = df['latency_mean_ms'].idxmin()
    fastest = df.loc[fastest_idx]
    report_lines.append(f"\n⚡ FASTEST CONFIGURATION:")
    report_lines.append(f"   Modality: {fastest['modality']}")
    report_lines.append(f"   Classifier: {fastest['classifier'].upper()}")
    report_lines.append(f"   Top-K: {fastest['top_k']}")
    report_lines.append(f"   Features: {fastest['n_features']}")
    report_lines.append(f"   Latency: {fastest['latency_mean_ms']:.3f} ± {fastest['latency_std_ms']:.3f} ms")

    # Average latency by classifier
    report_lines.append(f"\n📊 AVERAGE LATENCY BY CLASSIFIER:")
    clf_latency = df.groupby('classifier')['latency_mean_ms'].mean().sort_values()
    for clf, lat in clf_latency.items():
        report_lines.append(f"   {clf.upper()}: {lat:.3f} ms")

    # Average latency by Top-K
    report_lines.append(f"\n📈 AVERAGE LATENCY BY TOP-K:")
    # Sort top_k properly
    df['top_k_sort'] = df['top_k'].apply(lambda x: 999 if x == 'all' else int(x))
    topk_latency = df.groupby(['top_k', 'top_k_sort'])['latency_mean_ms'].mean().reset_index()
    topk_latency = topk_latency.sort_values('top_k_sort')
    for _, row in topk_latency.iterrows():
        report_lines.append(f"   Top-{row['top_k']}: {row['latency_mean_ms']:.3f} ms")

    # Best for tri-modality (clinical target)
    report_lines.append(f"\n🎯 TRI-MODALITY (ECG_EEG_Pupil) LATENCY:")
    tri_df = df[df['modality'] == 'ECG_EEG_Pupil'].copy()
    if not tri_df.empty:
        tri_df = tri_df.sort_values('top_k_sort')
        for _, row in tri_df.iterrows():
            report_lines.append(
                f"   Top-{row['top_k']} + {row['classifier'].upper()}: "
                f"{row['latency_mean_ms']:.3f} ms ({row['n_features']} features)"
            )

    # Latency vs Features correlation
    report_lines.append(f"\n📉 LATENCY VS FEATURE COUNT:")
    for mod in sorted(df['modality'].unique()):
        mod_df = df[df['modality'] == mod]
        # Get best classifier's latency progression
        best_clf = mod_df.groupby('classifier')['latency_mean_ms'].mean().idxmin()
        clf_df = mod_df[mod_df['classifier'] == best_clf].sort_values('top_k_sort')

        if len(clf_df) >= 2:
            min_lat = clf_df['latency_mean_ms'].min()
            max_lat = clf_df['latency_mean_ms'].max()
            min_feat = clf_df.loc[clf_df['latency_mean_ms'].idxmin(), 'n_features']
            max_feat = clf_df.loc[clf_df['latency_mean_ms'].idxmax(), 'n_features']
            report_lines.append(
                f"   {mod} ({best_clf.upper()}): "
                f"{min_feat} feat → {min_lat:.3f} ms | "
                f"{max_feat} feat → {max_lat:.3f} ms"
            )

    # Save report
    report_text = "\n".join(report_lines)
    report_path = os.path.join(output_dir, 'latency_summary.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)

    if verbose:
        print("\n" + report_text)
        print(f"\n✅ Report saved: {report_path}")


def generate_latency_plots(df: pd.DataFrame, output_dir: str, verbose: bool = True):
    """Generate latency visualization plots."""

    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 12

    fig_dir = os.path.join(output_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)

    # Ensure proper sorting
    df['top_k_sort'] = df['top_k'].apply(lambda x: 999 if x == 'all' else int(x))

    # 1. Latency by Classifier (Box Plot)
    plt.figure(figsize=(10, 6))
    clf_order = df.groupby('classifier')['latency_mean_ms'].mean().sort_values().index.tolist()

    data_for_box = [df[df['classifier'] == clf]['latency_mean_ms'].values for clf in clf_order]
    bp = plt.boxplot(data_for_box, labels=[c.upper() for c in clf_order], patch_artist=True)

    colors = plt.cm.Set2(np.linspace(0, 1, len(clf_order)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    plt.xlabel('Classifier', fontweight='bold', fontsize=12)
    plt.ylabel('Latency (ms)', fontweight='bold', fontsize=12)
    plt.title('Inference Latency by Classifier', fontweight='bold', fontsize=14)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()

    fig_path = os.path.join(fig_dir, 'latency_by_classifier.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()

    if verbose:
        print(f"✅ Saved: {fig_path}")

    # 2. Latency vs Top-K (Line Plot for Tri-modality)
    plt.figure(figsize=(12, 6))

    tri_df = df[df['modality'] == 'ECG_EEG_Pupil'].copy()
    if not tri_df.empty:
        for clf in sorted(tri_df['classifier'].unique()):
            clf_df = tri_df[tri_df['classifier'] == clf].sort_values('top_k_sort')
            plt.plot(
                range(len(clf_df)),
                clf_df['latency_mean_ms'],
                marker='o',
                label=clf.upper(),
                linewidth=2,
                markersize=8
            )

        top_k_labels = tri_df.sort_values('top_k_sort')['top_k'].unique()
        plt.xticks(range(len(top_k_labels)), [f"Top-{k}" for k in top_k_labels], fontweight='bold')
        plt.xlabel('Number of Features', fontweight='bold', fontsize=12)
        plt.ylabel('Latency (ms)', fontweight='bold', fontsize=12)
        plt.title('Inference Latency vs Feature Count (ECG+EEG+Pupil)', fontweight='bold', fontsize=14)
        plt.legend(loc='upper left', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        fig_path = os.path.join(fig_dir, 'latency_vs_topk_trimodal.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()

        if verbose:
            print(f"✅ Saved: {fig_path}")

    # 3. Latency Heatmap (Modality × Top-K for best classifier LGBM)
    lgbm_df = df[df['classifier'] == 'lgbm'].copy()
    if not lgbm_df.empty:
        pivot = lgbm_df.pivot_table(
            index='modality',
            columns='top_k',
            values='latency_mean_ms'
        )

        # Sort columns
        col_order = sorted(pivot.columns, key=lambda x: (x == 'all', int(x) if x != 'all' else 999))
        pivot = pivot[col_order]

        plt.figure(figsize=(14, 8))
        import seaborn as sns
        sns.heatmap(
            pivot,
            annot=True,
            fmt='.2f',
            cmap='YlOrRd',
            linewidths=0.5,
            cbar_kws={'label': 'Latency (ms)'},
            annot_kws={'fontsize': 10, 'fontweight': 'bold'}
        )
        plt.title('LGBM Inference Latency Heatmap (ms)', fontweight='bold', fontsize=14)
        plt.xlabel('Number of Features (Top-K)', fontweight='bold', fontsize=12)
        plt.ylabel('Modality', fontweight='bold', fontsize=12)
        plt.tight_layout()

        fig_path = os.path.join(fig_dir, 'latency_heatmap_lgbm.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()

        if verbose:
            print(f"✅ Saved: {fig_path}")

    # 4. Accuracy vs Latency Trade-off Plot (Need to load accuracy data)
    try:
        kfold_path = os.path.join(output_dir, 'ablation_stratified_kfold.csv')
        if os.path.exists(kfold_path):
            acc_df = pd.read_csv(kfold_path)

            # Merge with latency
            merged = df.merge(
                acc_df[['modality', 'classifier', 'top_k', 'accuracy_mean']],
                on=['modality', 'classifier', 'top_k'],
                how='inner'
            )

            # Filter to tri-modality only for cleaner plot
            tri_merged = merged[merged['modality'] == 'ECG_EEG_Pupil'].copy()

            if not tri_merged.empty:
                plt.figure(figsize=(12, 8))

                colors = {'xgb': 'red', 'rf': 'blue', 'et': 'green', 'lgbm': 'orange', 'cat': 'purple'}

                for clf in sorted(tri_merged['classifier'].unique()):
                    clf_data = tri_merged[tri_merged['classifier'] == clf]
                    plt.scatter(
                        clf_data['latency_mean_ms'],
                        clf_data['accuracy_mean'] * 100,
                        label=clf.upper(),
                        s=100,
                        c=colors.get(clf, 'gray'),
                        alpha=0.7,
                        edgecolors='black'
                    )

                    # Annotate with Top-K
                    for _, row in clf_data.iterrows():
                        plt.annotate(
                            f"T{row['top_k']}",
                            (row['latency_mean_ms'], row['accuracy_mean'] * 100),
                            textcoords="offset points",
                            xytext=(5, 5),
                            fontsize=8
                        )

                plt.xlabel('Latency (ms)', fontweight='bold', fontsize=12)
                plt.ylabel('Accuracy (%)', fontweight='bold', fontsize=12)
                plt.title('Accuracy vs Latency Trade-off (ECG+EEG+Pupil)', fontweight='bold', fontsize=14)
                plt.legend(loc='lower right', fontsize=10)
                plt.grid(True, alpha=0.3)

                # Add ideal region annotation
                plt.axhline(y=98, color='green', linestyle='--', alpha=0.5, label='98% Accuracy')
                plt.axvline(x=1.0, color='red', linestyle='--', alpha=0.5, label='1ms Latency')

                plt.tight_layout()

                fig_path = os.path.join(fig_dir, 'accuracy_vs_latency_tradeoff.png')
                plt.savefig(fig_path, dpi=300, bbox_inches='tight')
                plt.close()

                if verbose:
                    print(f"✅ Saved: {fig_path}")
    except Exception as e:
        if verbose:
            print(f"⚠️ Could not generate trade-off plot: {e}")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    print("\n" + "⚡" * 30)
    print("STARTING LATENCY BENCHMARK")
    print("⚡" * 30 + "\n")

    results = run_latency_benchmark(
        top_k_values=[10, 20, 30, 40, 50, 60, 70, 80, 'all'],
        classifiers=['xgb', 'rf', 'et', 'lgbm', 'cat'],
        n_repeat=100,
        verbose=True
    )

    print("\n" + "✅" * 30)
    print("LATENCY BENCHMARK COMPLETE!")
    print("✅" * 30)