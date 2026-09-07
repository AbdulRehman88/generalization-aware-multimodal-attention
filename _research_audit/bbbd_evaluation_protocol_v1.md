# BBBD Evaluation Protocol V1

Date: 2026-08-02

## Scientific target

The target is classification of the experimentally defined
attentive-versus-distracted viewing condition.

The condition remains perfectly confounded with session order, repeat exposure,
and the backward-counting dual task. Evaluation cannot isolate attention as the
sole causal source of physiological differences.

## Cohort

Direct seven-path comparison uses the identical participant-complete
multimodal cohort:

- 36 participants
- 392 recordings
- 20 Experiment 2 participants
- 16 Experiment 3 participants
- 51,831 accepted windows
- 196 attentive and 196 distracted recordings

## Primary within-experiment evaluation

Experiment 2 and Experiment 3 are evaluated separately using nested
participant-disjoint leave-one-participant-out evaluation.

For each outer fold:

1. One participant is reserved as the untouched outer test participant.
2. Remaining participants are divided deterministically into inner training and
   validation roles.
3. Cleanup, SHAP ranking, top-k selection, model selection, and threshold
   selection use only inner roles.
4. Outer-test data are accessed once after all selections are fixed.

A pooled-experiment LOSO result is not a primary analysis.

## Cross-experiment transfer

Two directions are prespecified:

- Experiment 2 to Experiment 3
- Experiment 3 to Experiment 2

Model and feature selection use the source experiment only. The target
experiment cannot influence feature cleanup, SHAP ranking, hyperparameters,
top-k selection, or threshold selection.

## Training weights

Each source recording contributes total fitting weight one. A window receives
weight equal to the inverse number of accepted windows in its recording.

No test-set balancing or test-label-dependent resampling is permitted.

## Aggregation and metrics

The primary decision unit is the recording. Recording probability is the
arithmetic mean of its window probabilities.

Primary metrics:

- recording-level balanced accuracy
- recording-level macro-F1

Secondary metrics include recording-level accuracy, ROC-AUC, PR-AUC, and
participant-macro recording metrics.

Window-level metrics are descriptive only.

## Uncertainty

Confidence intervals resample participants as clusters and retain all
recordings belonging to each sampled participant. The locked bootstrap count is
5,000 repetitions.

## Performance-chasing stop rule

The model families, hyperparameter grids, SHAP top-k candidates, threshold
candidates, metrics, aggregation policy, and uncertainty procedure are fixed
before examining outer-test or target-experiment performance.

No classifier or performance evaluation was run while creating this protocol.
