#!/usr/bin/env python3
import os, yaml, pandas as pd
from glob import glob

# ─── Configuration ────────────────────────────────────────────────────────────
groups = {
    'Single_modalities': ['ECG','EEG','Pupil'],
    'Dual_modalities':   ['ECG_EEG','ECG_Pupil','EEG_Pupil'],
    'Fused_modality':    ['ECG_EEG_Pupil']
}

cv_methods = [
    ('Stratified CV (10-fold)',   'stratified_kfold'),
    ('Repeated CV (5×10-fold)',   'repeated_kfold'),
    ('ShuffleSplit (100 splits)', 'shuffle_split'),
    ('Bootstrap (1000 iters)',     'bootstrap_ci'),
]

metrics       = ['accuracy','precision','recall','f1','roc_auc']
latex_metrics = ['Accuracy','Precision','Recall','F₁-score','ROC AUC']

def load_config():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__),'..','..'))
    with open(os.path.join(root,'configs','config.yaml'), encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    return cfg['summary']['output_dir']

def fmt_cell(df, method, m):
    """Format one cell:
       - for CV methods: mean±std
       - for bootstrap_ci: mean [lower,upper] for EVERY metric
    """
    if method != 'bootstrap_ci':
        mu = df.at[m,       'value']
        sd = df.at[f"{m}_std",'value']
        return f"{mu:.2f} ± {sd:.2f}"
    else:
        mn = df.at[m,           'value']
        lo = df.at[f"{m}_lower",'value']
        hi = df.at[f"{m}_upper",'value']
        return f"{mn:.2f} [{lo:.2f},{hi:.2f}]"

if __name__=="__main__":
    summary = load_config()

    for clf in ['XGB','RF','ET','LGBM','CAT']:
        for group_name, modalities in groups.items():
            out_dir = os.path.join(summary,'Tabular_CV_Results',group_name,clf)
            os.makedirs(out_dir, exist_ok=True)

            # Build rows for each (modality,metric)
            rows = []
            for mod in modalities:
                for m,lm in zip(metrics, latex_metrics):
                    row = [mod, lm]
                    for disp, cm in cv_methods:
                        csvf = os.path.join(summary, cm,
                                           f"metrics_{mod}_{clf.lower()}.csv")
                        df   = pd.read_csv(csvf).set_index('metric')
                        row.append(fmt_cell(df, cm, m))
                    rows.append(row)

            # Save CSV
            cols = ['Modality','Metric'] + [d for d,_ in cv_methods]
            df_out = pd.DataFrame(rows, columns=cols)
            csv_out = os.path.join(out_dir, f"table_{group_name}_{clf}.csv")
            df_out.to_csv(csv_out, index=False)
            print("Saved CSV →", csv_out)

            # Build LaTeX
            tex = []
            tex.append(r"% requires: \usepackage{booktabs,multirow}")
            tex.append(r"\begin{tabular}{lccccc}")
            tex.append(r"\toprule")
            tex.append(" & ".join(cols) + r" \\")
            tex.append(r"\midrule")
            for i, mod in enumerate(modalities):
                block = df_out[df_out.Modality==mod]
                for j in range(5):
                    parts = []
                    if j==0:
                        parts.append(rf"\multirow{{5}}{{*}}{{{mod.replace('_','+')}}}")
                    else:
                        parts.append("")
                    parts.append(block.iloc[j].Metric)
                    for disp,_ in cv_methods:
                        parts.append(block.iloc[j][disp])
                    tex.append(" & ".join(parts) + r" \\")
                if group_name!='Fused_modality' and i < len(modalities)-1:
                    tex.append(r"\midrule")
            tex.append(r"\bottomrule")
            tex.append(r"\end{tabular}")

            tex_out = os.path.join(out_dir, f"table_{group_name}_{clf}.tex")
            with open(tex_out, "w", encoding="utf-8") as f:
                f.write("\n".join(tex))
            print("Saved LaTeX →", tex_out)
