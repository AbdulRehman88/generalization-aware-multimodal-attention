import os
import yaml

# 1) feature extraction + fusion
from src.features.extract_features       import main as extract_features
# 2) SHAP selection
from src.shap_selection.select_features  import select_features
# 3) training
from src.models.train_models             import train_for_modality
# 4) evaluation plots
from src.evaluation.plotting             import plot_confusion, plot_roc
# 5) latency benchmarking
from src.evaluation.benchmark_latency    import run_benchmark

def load_config(path="configs/config.yaml"):
    # force UTF-8 decoding to avoid cp949 errors on Windows
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    cfg = load_config()

    # 1) Extract & fuse features
    print("=== Extracting & fusing features ===")
    extract_features()

    # 2) SHAP feature selection
    print("\n=== Selecting SHAP features ===")
    select_features("configs/config.yaml", top_k=20)

    # 3) Train models on every modality
    print("\n=== Training models ===")
    for mod in cfg['modalities']:
        print(f"\n--- {mod} ---")
        train_for_modality(mod, cfg, top_k=20)

    # 4) Generate confusion matrices & ROC curves
    print("\n=== Generating evaluation plots ===")
    os.makedirs(cfg["figures"]["output_dir"], exist_ok=True)
    classifiers = ['xgb','rf','et','lgbm','cat']
    for mod in cfg['modalities']:
        for clf in classifiers:
            plot_confusion(mod, clf, cfg, top_k=20)
            plot_roc(mod, clf, cfg, top_k=20)

    # 5) Latency benchmarking (if enabled in config.yaml)
    if cfg.get("evaluation", {}).get("benchmark", {}).get("enable", False):
        print("\n=== Running latency benchmark ===")
        run_benchmark(cfg)
        print("=== Benchmarking complete ===")

if __name__ == "__main__":
    main()
