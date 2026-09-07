# BBBD Fold-Aware Results Final V1

Date: 2026-08-03

## Evaluation identity

This package reports classification of the experimentally defined
attentive-versus-distracted viewing condition in BBBD.

The labels must not be interpreted as an isolated causal measure of attention.
Condition is perfectly confounded with fixed session order, repeat exposure,
and the backward-counting dual task. Fatigue, day, and device drift may also
contribute to the observed physiological differences.

## Verified cohort and execution

- Participants: 36
- Recordings: 392
- Accepted four-second windows: 51831
- Experiment 2 participants: 20
- Experiment 3 participants: 16
- Balanced recording labels: 196 distracted and 196 attentive
- Seven modality paths
- 252 within-experiment path/fold evaluations
- 14 cross-experiment path/direction evaluations
- 266 completed path/split evaluations
- 28 fold-aware result groups
- 5,000 participant-cluster bootstrap repetitions per group
- No classifier or selector was refitted during reporting

Cohort counts were verified directly from the locked trimodal feature metadata,
rather than inferred from an optional summary-field name.

## Metric policy

Balanced accuracy and macro-F1 are the primary recording-level metrics.

Threshold-dependent metrics use each split's validation-selected threshold.
ROC-AUC and PR-AUC use untouched out-of-fold or target-experiment
probabilities. Window metrics are descriptive only.

## Best path by locked primary order

| protocol | evaluation | path | recording_balanced_accuracy | recording_macro_f1 | recording_roc_auc | recording_pr_auc |
|---|---|---|---|---|---|---|
| cross_experiment | experiment2_to_experiment3 | EEG | 0.6458 | 0.6222 | 0.6611 | 0.6298 |
| cross_experiment | experiment3_to_experiment2 | ECG_EEG_Pupil | 0.6150 | 0.6098 | 0.6238 | 0.5885 |
| within_experiment | experiment2 | ECG_EEG_Pupil | 0.6000 | 0.5960 | 0.6798 | 0.6593 |
| within_experiment | experiment3 | ECG_EEG_Pupil | 0.6302 | 0.6301 | 0.6760 | 0.6473 |

Trimodal ECG+EEG+pupil fusion ranks first in three evaluations. EEG alone ranks
first for Experiment 2 to Experiment 3 transfer. These rankings are
evaluation-specific and do not establish universal modality superiority.

## Main evidence pattern

The evidence shows moderate experimental-condition discrimination:

- Within Experiment 2, the best recording balanced accuracy is 0.6000.
- Within Experiment 3, the best recording balanced accuracy is 0.6302.
- Experiment 2 to Experiment 3 transfer reaches 0.6458 with EEG.
- Experiment 3 to Experiment 2 transfer reaches 0.6150 with trimodal fusion.
- ECG alone remains approximately chance-level.
- Pupil features provide useful probability ranking in some settings, including
  ROC-AUC 0.7461 within Experiment 3.
- Participant-macro F1 is frequently lower than pooled recording macro-F1,
  indicating meaningful participant-level heterogeneity.

## Complete recording-level results

| protocol | evaluation | path | recording_balanced_accuracy | recording_macro_f1 | recording_roc_auc | recording_pr_auc | participant_macro_f1 |
|---|---|---|---|---|---|---|---|
| cross_experiment | experiment2_to_experiment3 | ECG | 0.5052 | 0.4901 | 0.5188 | 0.5258 | 0.3761 |
| cross_experiment | experiment2_to_experiment3 | ECG_EEG | 0.6302 | 0.5997 | 0.6770 | 0.6382 | 0.5321 |
| cross_experiment | experiment2_to_experiment3 | ECG_EEG_Pupil | 0.6354 | 0.6132 | 0.6552 | 0.6528 | 0.5505 |
| cross_experiment | experiment2_to_experiment3 | ECG_Pupil | 0.5521 | 0.5520 | 0.5611 | 0.5780 | 0.4253 |
| cross_experiment | experiment2_to_experiment3 | EEG | 0.6458 | 0.6222 | 0.6611 | 0.6298 | 0.5533 |
| cross_experiment | experiment2_to_experiment3 | EEG_Pupil | 0.5938 | 0.5733 | 0.6866 | 0.6806 | 0.5053 |
| cross_experiment | experiment2_to_experiment3 | Pupil | 0.5625 | 0.5152 | 0.7063 | 0.6807 | 0.4476 |
| cross_experiment | experiment3_to_experiment2 | ECG | 0.4800 | 0.4724 | 0.4900 | 0.5120 | 0.3570 |
| cross_experiment | experiment3_to_experiment2 | ECG_EEG | 0.5700 | 0.5647 | 0.6142 | 0.5703 | 0.4660 |
| cross_experiment | experiment3_to_experiment2 | ECG_EEG_Pupil | 0.6150 | 0.6098 | 0.6238 | 0.5885 | 0.5108 |
| cross_experiment | experiment3_to_experiment2 | ECG_Pupil | 0.5550 | 0.5313 | 0.5939 | 0.6099 | 0.4481 |
| cross_experiment | experiment3_to_experiment2 | EEG | 0.6050 | 0.6045 | 0.6058 | 0.5941 | 0.5330 |
| cross_experiment | experiment3_to_experiment2 | EEG_Pupil | 0.5450 | 0.5323 | 0.6176 | 0.6196 | 0.4208 |
| cross_experiment | experiment3_to_experiment2 | Pupil | 0.6150 | 0.5998 | 0.6762 | 0.6695 | 0.5210 |
| within_experiment | experiment2 | ECG | 0.5250 | 0.4891 | 0.5214 | 0.5524 | 0.3892 |
| within_experiment | experiment2 | ECG_EEG | 0.5500 | 0.5366 | 0.5527 | 0.5633 | 0.4239 |
| within_experiment | experiment2 | ECG_EEG_Pupil | 0.6000 | 0.5960 | 0.6798 | 0.6593 | 0.4810 |
| within_experiment | experiment2 | ECG_Pupil | 0.5150 | 0.5014 | 0.6119 | 0.5965 | 0.3835 |
| within_experiment | experiment2 | EEG | 0.5750 | 0.5749 | 0.6496 | 0.6487 | 0.4656 |
| within_experiment | experiment2 | EEG_Pupil | 0.5900 | 0.5896 | 0.6524 | 0.6425 | 0.4753 |
| within_experiment | experiment2 | Pupil | 0.5950 | 0.5938 | 0.6555 | 0.6209 | 0.5028 |
| within_experiment | experiment3 | ECG | 0.5000 | 0.4749 | 0.4898 | 0.5045 | 0.3713 |
| within_experiment | experiment3 | ECG_EEG | 0.6250 | 0.6240 | 0.6704 | 0.6803 | 0.5169 |
| within_experiment | experiment3 | ECG_EEG_Pupil | 0.6302 | 0.6301 | 0.6760 | 0.6473 | 0.5361 |
| within_experiment | experiment3 | ECG_Pupil | 0.5833 | 0.5804 | 0.6304 | 0.6052 | 0.4819 |
| within_experiment | experiment3 | EEG | 0.5260 | 0.5203 | 0.5969 | 0.5934 | 0.4007 |
| within_experiment | experiment3 | EEG_Pupil | 0.6146 | 0.6145 | 0.6748 | 0.6523 | 0.5237 |
| within_experiment | experiment3 | Pupil | 0.6094 | 0.6037 | 0.7461 | 0.7675 | 0.5219 |

## Supported claims

The evidence supports leakage-safe experimental-condition classification,
participant-disjoint within-experiment evaluation, independent-cohort
cross-experiment transfer under domain shift, and aligned seven-path modality
comparison.

The evidence does not support unconfounded attention recognition, causal
attribution to attention alone, universal trimodal superiority, real-world
deployment effectiveness, or external clinical or operational validity.

## Reproducibility

The accompanying JSON contains all 28 results, cohort counts derived directly
from locked feature metadata, confidence intervals, selection distributions,
scientific claim boundaries, and hashes for the data evidence, configuration,
protocol, runner, reporting implementation, and report outputs.
