#!/usr/bin/env python3
import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob

# ─── Style ───────────────────────────────────────────────────────────────────
plt.rc('font', family='Times New Roman', size=12)
BAR_COLORS   = plt.get_cmap('Pastel1').colors
ROC_COLORS   = ['tab:blue','tab:orange','tab:green']  # Low, Mid, High

MOD_ORDER    = ['ECG','EEG','Pupil','ECG_EEG','ECG_Pupil','EEG_Pupil','ECG_EEG_Pupil']
CLASSIFIERS  = ['XGB','RF','ET','LGBM','CAT']
CLASS_LABELS = ['Low Attention','Mid Attention','High Attention']

def load_summary_dir():
    root  = os.path.abspath(os.path.join(os.path.dirname(__file__),'..','..'))
    cfg_p = os.path.join(root,'configs','config.yaml')
    with open(cfg_p, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    return cfg['summary']['output_dir']

def parse_name(fname, prefix):
    base,_ = os.path.splitext(os.path.basename(fname))
    p,rest = base.split('_',1)
    if p!=prefix:
        raise ValueError(f"Expected '{prefix}_…', got '{base}'")
    mod,clf = rest.rsplit('_',1)
    return mod, clf.upper()

def visualize_results():
    summary = load_summary_dir()
    n_mod   = len(MOD_ORDER)
    x       = np.arange(n_mod)

    for method in ['stratified_kfold','repeated_kfold','shuffle_split','bootstrap_ci']:
        title_m = method.replace('_',' ').title()
        eval_dir= os.path.join(summary, method)
        fig_dir = os.path.join(eval_dir, 'figures')
        os.makedirs(fig_dir, exist_ok=True)
        print(f"\n=== Visualizing {method} ===")

        # 1) Bar charts
        if method!='bootstrap_ci':
            for metric in ['accuracy','precision','recall','f1']:
                records = []
                for csvf in sorted(glob(os.path.join(eval_dir,'metrics_*.csv'))):
                    mod,clf = parse_name(csvf,'metrics')
                    df = pd.read_csv(csvf).set_index('metric')
                    mean = df.at[metric,'value']
                    std  = df.at[f"{metric}_std",'value']
                    records.append((mod,clf,mean,std))
                dfm = pd.DataFrame(records, columns=['modality','classifier','mean','std'])
                dfm = dfm.set_index(['modality','classifier']).unstack('classifier')
                means = dfm['mean'].reindex(MOD_ORDER)[CLASSIFIERS].values
                errs  = dfm['std'].reindex(MOD_ORDER)[CLASSIFIERS].values

                width = 0.8/len(CLASSIFIERS)
                fig, ax = plt.subplots(figsize=(10,6))
                for i,clf in enumerate(CLASSIFIERS):
                    ax.bar(
                        x + (i - 2)*width,
                        means[:,i],
                        width,
                        yerr=errs[:,i],
                        capsize=4,
                        color=BAR_COLORS[i],
                        edgecolor='black',
                        label=clf
                    )
                ax.set_title(f"{title_m} – {metric.capitalize()}", fontweight='bold')
                ax.set_xlabel('Modality', fontweight='bold')
                ax.set_ylabel(metric.capitalize(), fontweight='bold')
                ax.set_xticks(x)
                ax.set_xticklabels(MOD_ORDER, rotation=45, ha='right')
                ax.set_ylim(0,1.02)
                ax.tick_params(labelsize=12)
                leg = ax.legend(title='Classifier', loc='upper left',
                                bbox_to_anchor=(1.02,1), frameon=False,
                                prop={'size':12,'weight':'bold'})
                for txt in leg.get_texts(): txt.set_fontweight('bold')
                plt.tight_layout()
                plt.savefig(os.path.join(fig_dir,f"{metric}.png"), dpi=300, bbox_inches='tight')
                plt.close(fig)

        else:
            # bootstrap: loop all four metrics with 95% CI
            for metric in ['accuracy','precision','recall','f1']:
                records = []
                for csvf in sorted(glob(os.path.join(eval_dir,'metrics_*.csv'))):
                    mod,clf = parse_name(csvf,'metrics')
                    df = pd.read_csv(csvf).set_index('metric')
                    mn = df.at[metric,           'value']
                    lo = df.at[f"{metric}_lower",'value']
                    hi = df.at[f"{metric}_upper",'value']
                    records.append((mod,clf,mn, mn-lo, hi-mn))
                dfb = pd.DataFrame(records, columns=['modality','classifier','mean','err_lo','err_hi'])
                dfb = dfb.set_index(['modality','classifier']).unstack('classifier')
                means   = dfb['mean'].reindex(MOD_ORDER)[CLASSIFIERS].values
                errs_lo = dfb['err_lo'].reindex(MOD_ORDER)[CLASSIFIERS].values
                errs_hi = dfb['err_hi'].reindex(MOD_ORDER)[CLASSIFIERS].values

                width = 0.8/len(CLASSIFIERS)
                fig, ax = plt.subplots(figsize=(10,6))
                for i,clf in enumerate(CLASSIFIERS):
                    ax.bar(
                        x + (i - 2)*width,
                        means[:,i],
                        width,
                        yerr=[errs_lo[:,i], errs_hi[:,i]],
                        capsize=4,
                        color=BAR_COLORS[i],
                        edgecolor='black',
                        label=clf
                    )
                ax.set_title(f"{title_m} – {metric.capitalize()} (bootstrap 95% CI)", fontweight='bold')
                ax.set_xlabel('Modality', fontweight='bold')
                ax.set_ylabel(metric.capitalize(), fontweight='bold')
                ax.set_xticks(x)
                ax.set_xticklabels(MOD_ORDER, rotation=45, ha='right')
                ax.set_ylim(0,1.02)
                ax.tick_params(labelsize=12)
                leg = ax.legend(title='Classifier', loc='upper left',
                                bbox_to_anchor=(1.02,1), frameon=False,
                                prop={'size':12,'weight':'bold'})
                for txt in leg.get_texts(): txt.set_fontweight('bold')
                plt.tight_layout()
                plt.savefig(os.path.join(fig_dir, f"{metric}.png"), dpi=300, bbox_inches='tight')
                plt.close(fig)

        # 2) ROC curves
        for npzf in sorted(glob(os.path.join(eval_dir,'roc_*.npz'))):
            mod,clf = parse_name(npzf,'roc')
            arr = np.load(npzf)

            fig, ax = plt.subplots(figsize=(8,6))
            if all(f"fpr_{i}" in arr for i in range(3)):
                for i,label in enumerate(CLASS_LABELS):
                    ax.plot(arr[f"fpr_{i}"], arr[f"tpr_{i}"],
                            lw=2, color=ROC_COLORS[i],
                            label=f"{label} (AUC={arr[f'auc_{i}']:.2f})")
            else:
                ax.plot(arr['fpr'], arr['tpr'], lw=2,
                        color='black', label=f"AUC={arr['auc']:.2f}")

            ax.plot([0,1],[0,1],'--',color='gray',lw=1)
            ax.set_title(f"{title_m} – {mod}_{clf} ROC Curve", fontweight='bold')
            ax.set_xlabel('False Positive Rate', fontweight='bold')
            ax.set_ylabel('True Positive Rate', fontweight='bold')
            ax.tick_params(labelsize=12)
            leg = ax.legend(loc='lower right', frameon=False,
                            prop={'size':11,'weight':'bold'})
            for txt in leg.get_texts(): txt.set_fontweight('bold')
            plt.tight_layout()
            plt.savefig(os.path.join(fig_dir,f"roc_{mod}_{clf}.png"), dpi=300, bbox_inches='tight')
            plt.close(fig)

        # 3) Confusion matrices
        for npzf in sorted(glob(os.path.join(eval_dir,'roc_*.npz'))):
            arr = np.load(npzf)
            if 'cm' not in arr: continue
            mod,clf = parse_name(npzf,'roc')
            cm = arr['cm'].astype(int)

            fig, ax = plt.subplots(figsize=(6,6))
            sns.heatmap(
                cm, annot=True, fmt='d', cmap='Blues',
                cbar=False, linewidths=1, linecolor='white',
                annot_kws={'size':14,'weight':'bold'},
                xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS,
                ax=ax
            )
            ax.set_title(f"{title_m} – {mod}_{clf} Confusion Matrix", fontweight='bold')
            ax.set_xlabel('Predicted', fontweight='bold')
            ax.set_ylabel('True',      fontweight='bold')
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontweight='bold')
            plt.setp(ax.get_yticklabels(), rotation=0,              fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(fig_dir,f"cm_{mod}_{clf}.png"), dpi=300, bbox_inches='tight')
            plt.close(fig)

    print("\nPhase 2 complete: all figures saved.")

if __name__=="__main__":
    visualize_results()
