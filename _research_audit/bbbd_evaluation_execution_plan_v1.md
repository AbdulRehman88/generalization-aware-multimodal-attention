# BBBD Evaluation Execution Plan V1

Date: 2026-08-02

## Status

This execution plan was generated before fitting a BBBD feature selector or
classifier and before observing outer-test or target-experiment performance.

- Classifier fitted: no
- SHAP selector fitted: no
- Performance observed: no
- Source protocol commit: `17ab03d`

## Evaluation structure

- Separate within-experiment nested LOSO:
  - Experiment 2: 20 outer folds
  - Experiment 3: 16 outer folds
- Cross-experiment transfer:
  - Experiment 2 to Experiment 3
  - Experiment 3 to Experiment 2
- Target-experiment tuning: prohibited
- Direct modality paths: 7

## Locked model families

Two fixed model configurations are retained:

- Extra Trees: 500 estimators, unrestricted depth, minimum leaf size 1,
  square-root feature sampling
- XGBoost: 400 estimators, depth 3, learning rate 0.05, row and feature
  subsampling 0.8

Model family and effective SHAP top-k are selected using validation
participants only. No outer-test result may alter these settings.

## Effective top-k candidates

Configured candidates are 20, 50, 100, and all. Values exceeding a modality's
feature dimension are collapsed to the same effective feature set.

- ECG: [16]
- Pupil: [20, 23]
- ECG+Pupil: [20, 39]
- EEG: [20, 50, 100, 232]
- ECG+EEG: [20, 50, 100, 248]
- EEG+Pupil: [20, 50, 100, 255]
- ECG+EEG+Pupil: [20, 50, 100, 271]

Candidate counts per evaluation split are:

- ECG: 2
- Pupil: 4
- ECG+Pupil: 4
- EEG: 8
- ECG+EEG: 8
- EEG+Pupil: 8
- ECG+EEG+Pupil: 8

The seven paths therefore require 42 validation candidate fits per outer fold
or cross-experiment direction.

## Locked fit budget

- Within-experiment outer folds: 36
- Cross-experiment directions: 2
- Candidate model fits: 1596
- SHAP selector fits: 266
- Selected-candidate final refits: 266
- Total estimator fits: 2128

## Scientific rationale for grid compaction

The original broad Cartesian grid would have introduced thousands of
scientifically redundant fits, including multiple configured top-k values that
produce identical feature subsets in low-dimensional ECG and pupil paths.

The grid was compacted before any BBBD performance was observed. The revision
retains:

- both prespecified model families;
- four meaningful SHAP subset scales where dimensionality permits;
- inner-participant model and subset selection;
- recording-equal fitting weights;
- recording-level threshold selection;
- untouched outer participants and target experiments.

This execution plan is now frozen. Performance results may not be used to
expand, replace, or retune it.
