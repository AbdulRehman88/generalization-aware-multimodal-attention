# Generalization-Aware Multimodal Physiological Monitoring in XR

Open-source implementation and reproducibility materials for:

> **Generalization-Aware Multimodal Physiological Monitoring in XR with SHAP-Guided Modular Learning Across Users and Sensor Configurations**  
> Abdul Rehman and Sungchul Mun

The framework integrates electroencephalography (EEG), electrocardiography (ECG), and pupil dynamics through seven unimodal, bimodal, and trimodal paths. It combines corrected physiological feature extraction, SHAP-guided feature selection, participant-aware evaluation, subject adaptation, raw-signal deep-learning baselines, external-dataset benchmarking, and latency analysis.

## Scientific scope

The evaluation hierarchy deliberately separates within-dataset discriminative capacity from deployment-relevant generalization:

1. conventional internal evaluations for comparability;
2. strict calibration-free nested leave-one-participant-out (LOSO) evaluation as the primary internal unseen-user protocol;
3. participant-adaptive evaluation with 30, 60, and 120 seconds of labeled calibration data per class;
4. participant-disjoint BBBD benchmarking within Experiments 2 and 3;
5. bidirectional BBBD cross-experiment transfer;
6. temporal-window sensitivity; and
7. ShallowConvNet and multibranch temporal convolutional network baselines.

The seven supported sensor configurations are EEG, ECG, Pupil, EEG+ECG, EEG+Pupil, ECG+Pupil, and EEG+ECG+Pupil.

## Headline results

- Conventional internal trimodal Top-20 accuracy: **97.18%**.
- Primary strict calibration-free nested LOSO: **70.84% accuracy**, **67.57% balanced accuracy**, and **67.01% macro-F1**.
- Participant-adaptive balanced accuracy at 30/60/120 seconds per class: **81.83% / 72.07% / 85.35%**.
- Best engineered-feature BBBD balanced accuracy within Experiments 2 and 3: **60.00% / 63.02%**.
- Best engineered-feature BBBD cross-experiment balanced accuracy for Experiment 2→3 and Experiment 3→2: **64.58% / 61.50%**.

These values must be interpreted under their stated protocols. The BBBD outcome represents attentive-versus-distracted experimental conditions and is not treated as a universal latent-attention label.

## Repository layout

```text
configs/                 Public and executed protocol configurations
src/                     Preprocessing, features, models, and evaluation code
tests/                   Unit and contract tests
artifacts/.../manifests/ Public BBBD cohort and recording manifests
_research_audit/         Public BBBD audit evidence
results/aggregate/       Aggregate internal and external result tables
figures/manuscript/      Corrected manuscript figures
supplementary/           Supplementary tables and source
```

## Installation

The executed environment used Python 3.10.11. On Windows PowerShell:

```powershell
git clone https://github.com/AbdulRehman88/generalization-aware-multimodal-attention.git
cd generalization-aware-multimodal-attention
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt` installs the CUDA 12.6 build used for the deep-learning experiments. `environment.lock.txt` records the complete executed Python environment. For CPU-only systems, install an appropriate PyTorch build following the official PyTorch installation instructions, then install the remaining requirements.

## Configuration

Copy the machine-local template and edit only the paths:

```powershell
Copy-Item configs\config.local.example.yaml configs\config.local.yaml
```

The audited runtime merges `configs/config.release.yaml` with the ignored machine-local file. Relative paths are resolved from the repository root.

Validate the configuration:

```powershell
python -m src.core.config --check-inputs
```

The legacy application path uses the portable compatibility file `configs/config.yaml`. Run legacy commands from the repository root.

## External BBBD data

The external analysis uses Experiments 2 and 3 of the Brain, Body, and Behavior Dataset (BBBD):

- Dataset DOI: https://doi.org/10.15387/fcp_indi.retro.bbbd
- Data record: https://zenodo.org/records/19241964
- Data paper: https://doi.org/10.1038/s41597-026-07215-1

Download `experiment2.zip` and `experiment3.zip`, keep the archives unchanged, and set their locations in `configs/config.local.yaml`. The repository does not redistribute the third-party raw archives.

Inspect available entry-point options before a long run:

```powershell
python -m src.evaluation.bbbd_runner --help
python -m src.evaluation.deep_learning_real_training_v3 --help
python -m src.evaluation.internal_nested_loso_long_windows --help
```

## Tests

```powershell
python -m unittest discover -s tests -p "test_*.py"
```

Tests that require restricted internal artifacts or multi-gigabyte generated caches are skipped when those resources are not installed. All synthetic, configuration, architecture, and distributed external-evidence tests remain executable.

## Data availability and privacy

The internal XR dataset contains participant-level physiological recordings and is not distributed because of privacy and ethical restrictions. This repository includes the processing and evaluation code but excludes internal raw/processed signals, features, predictions, participant-level metrics, split records, and model checkpoints. Only aggregate internal results reported in the manuscript are included.

The BBBD-derived public materials include code, cohort manifests, aggregate results, and audit evidence. Users must obtain the original BBBD archives from the dataset provider and comply with its CC BY 4.0 terms.

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the exact reproducibility boundary.

## Citation

Use the metadata in `CITATION.cff`. A version-specific DOI will be added after the GitHub release is archived by Zenodo.

## License

The software is released under the BSD 3-Clause License. External datasets and third-party packages retain their own licenses.

