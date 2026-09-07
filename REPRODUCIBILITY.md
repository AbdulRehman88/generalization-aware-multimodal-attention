# Reproducibility statement

## Publicly reproducible components

- installation of the complete software environment;
- execution of the unit and protocol-contract tests that do not require restricted artifacts;
- BBBD manifest construction, preprocessing, feature extraction, participant-disjoint evaluation, cross-experiment transfer, temporal sensitivity, and deep-learning analyses after obtaining the public BBBD archives;
- regeneration and verification of the distributed aggregate tables and figures;
- verification of published BBBD provenance and audit manifests.

## Restricted internal dataset

The 13-participant internal XR dataset is not publicly distributed. The public repository therefore excludes raw and processed internal signals, window-level features, participant-level predictions and metrics, calibration/test split records, cached tensors, and trained checkpoints.

The internal code and protocol definitions are public. Authorized users with the internal dataset can configure its paths locally and execute the same pipeline. Other users can inspect the aggregate results but cannot independently regenerate those internal values from the repository alone.

## Evaluation hierarchy

Strict calibration-free nested LOSO is the primary internal unseen-participant evaluation. Participant-specific calibration results are separate adaptive-use cases. Conventional grouped and window-level evaluations are comparability or in-distribution capacity analyses and must not be described as equivalent evidence of new-user generalization.

BBBD within-experiment analyses are participant-disjoint external-dataset benchmarks. BBBD bidirectional cross-experiment analyses measure transfer under combined cohort and experimental-domain shift. They are not frozen transfer of the internal three-class classifier.

## Determinism and hardware

The executed environment used Python 3.10.11 and PyTorch 2.10.0 with CUDA 12.6 on an NVIDIA RTX 4090. Fixed seeds and frozen protocol registries control model initialization and evaluation roles. Small numerical differences may remain across operating systems, CPU instruction sets, GPU architectures, drivers, and compiled library builds.

