# BBBD Independent Results Verification V1

Date: 2026-08-03

## Status

The committed BBBD evidence at source commit `622cedc` was independently
recomputed from the stored fold-aware recording predictions.

No feature selector or classifier was fitted during this verification.

## Verification coverage

- Result groups independently recomputed: 28
- Stored participant-bootstrap rows checked: 140000
- Bootstrap repetitions per result group: 5,000
- Evidence file hashes verified: 156
- Maximum absolute numerical difference: 2.220e-16

## Independently verified components

- pooled recording accuracy, balanced accuracy, macro-F1, ROC-AUC, and PR-AUC;
- participant-specific recording metrics;
- participant-macro metrics;
- task-specific recording metrics;
- confidence intervals recomputed from every stored bootstrap table;
- committed CSV versus source reporting summary;
- committed JSON versus committed CSV;
- model, effective-top-k, and threshold selection distributions;
- all evidence-package file hashes.

## Best paths under the locked primary ordering

- cross_experiment / experiment2_to_experiment3: EEG, balanced accuracy 0.6458, macro-F1 0.6222
- cross_experiment / experiment3_to_experiment2: ECG_EEG_Pupil, balanced accuracy 0.6150, macro-F1 0.6098
- within_experiment / experiment2: ECG_EEG_Pupil, balanced accuracy 0.6000, macro-F1 0.5960
- within_experiment / experiment3: ECG_EEG_Pupil, balanced accuracy 0.6302, macro-F1 0.6301

## Scientific interpretation

The independently verified results support moderate classification of the
experimentally defined attentive-versus-distracted condition under
participant-disjoint and cross-experiment evaluation.

They do not establish isolated or unconfounded attention recognition because
condition remains confounded with fixed session order, repeat exposure, and
backward counting, with possible fatigue, day, and device drift.

The results also do not establish universal trimodal superiority or deployment
validity.

## Decision

The committed final BBBD evidence is internally consistent, hash-valid, and
numerically reproducible from the stored predictions and bootstrap artifacts.
