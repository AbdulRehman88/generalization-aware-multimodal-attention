# File: src/evaluation/sanity_check_models.py

import os
import glob
import sys

# Add project root so we can import config.yaml if needed
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import yaml

def load_config(path="configs/config.yaml"):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)

def parse_model_filename(fname: str):
    """
    Given 'ecg_eeg_xgb.pkl' → ('ECG_EEG', 'XGB'). All uppercase.
    """
    base = os.path.basename(fname).replace(".pkl", "")
    parts = base.split("_")
    clf = parts[-1].upper()
    modality = "_".join(parts[:-1]).upper()
    return modality, clf

if __name__ == "__main__":
    cfg = load_config()
    model_dir = cfg["models"]["output_dir"]
    print("Model directory:", model_dir)
    all_models = sorted(glob.glob(os.path.join(model_dir, "*.pkl")))
    print(f"\nFound {len(all_models)} .pkl files:\n")

    allowed_modalities = set(m.upper() for m in cfg["modalities"])
    allowed_classifiers = {"XGB", "RF", "ET", "LGBM", "CAT"}  # exactly those 5

    valid = []
    invalid = []

    for path in all_models:
        modality, clf = parse_model_filename(path)
        flag_mod = modality in allowed_modalities
        flag_clf = clf in allowed_classifiers
        reason = []
        if not flag_mod:
            reason.append(f"bad-modality({modality})")
        if not flag_clf:
            reason.append(f"bad-classifier({clf})")
        if flag_mod and flag_clf:
            valid.append((modality, clf, path))
        else:
            invalid.append((modality, clf, path, ";".join(reason)))

    print(">>> VALID MODELS (should be 7×5 = 35) <<<")
    for mod, clf, p in valid:
        print(f"  {mod}+{clf}   {os.path.basename(p)}")
    print(f"  → TOTAL VALID: {len(valid)}\n")

    print(">>> INVALID / SKIPPED MODELS <<<")
    for mod, clf, p, rsn in invalid:
        print(f"  {mod}+{clf}   {os.path.basename(p)}   [{rsn}]")
    print(f"  → TOTAL SKIPPED: {len(invalid)}\n")
