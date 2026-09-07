# BBBD Matched Thirty-Two-Second Numerical Evidence V1

Date: 2026-08-06

## Verdict

All numerical values in the 28 matched thirty-two-second reports were independently recomputed from recording-level predictions.

The audit found zero mismatches across pooled metrics, participant-level metrics, participant-macro summaries, validation-selected threshold decisions, 5,000-repetition participant-cluster confidence intervals, and report registry and manifest hashes.

## Scientific role

Thirty-two seconds is a prespecified post-primary temporal-sensitivity duration. It does not replace the immutable primary four-second analysis on 392 recordings or the matched four-second sensitivity reference on 387 recordings.

These BBBD results measure attentive-versus-distracted experimental-condition classification. They do not isolate attention from session order, repeat exposure, fatigue, day/device drift, or the backward-counting dual task.

## Complete recording-level results

| Analysis | Setting | Path | Balanced accuracy | Macro-F1 | ROC-AUC | PR-AUC | Participant macro-F1 | Descriptive rank |
|---|---|---|---:|---:|---:|---:|---:|---:|
| cross_experiment | experiment2_to_experiment3 | EEG_Pupil | 0.650515 | 0.646585 | 0.671281 | 0.640214 | 0.566821 | 1 |
| cross_experiment | experiment2_to_experiment3 | EEG | 0.644565 | 0.637085 | 0.677231 | 0.660691 | 0.567523 | 2 |
| cross_experiment | experiment2_to_experiment3 | ECG_EEG_Pupil | 0.640160 | 0.637565 | 0.702746 | 0.726048 | 0.544198 | 3 |
| cross_experiment | experiment2_to_experiment3 | ECG_EEG | 0.572597 | 0.572082 | 0.643936 | 0.632441 | 0.485870 | 4 |
| cross_experiment | experiment2_to_experiment3 | ECG_Pupil | 0.532380 | 0.514286 | 0.626831 | 0.663162 | 0.381849 | 5 |
| cross_experiment | experiment2_to_experiment3 | Pupil | 0.519508 | 0.434068 | 0.615103 | 0.614292 | 0.386882 | 6 |
| cross_experiment | experiment2_to_experiment3 | ECG | 0.471568 | 0.469070 | 0.490732 | 0.540190 | 0.369339 | 7 |
| cross_experiment | experiment3_to_experiment2 | EEG_Pupil | 0.635000 | 0.634991 | 0.649600 | 0.668252 | 0.546575 | 1 |
| cross_experiment | experiment3_to_experiment2 | EEG | 0.610000 | 0.588217 | 0.673000 | 0.645451 | 0.509011 | 2 |
| cross_experiment | experiment3_to_experiment2 | Pupil | 0.600000 | 0.563271 | 0.673100 | 0.666203 | 0.503750 | 3 |
| cross_experiment | experiment3_to_experiment2 | ECG_EEG | 0.580000 | 0.546974 | 0.721600 | 0.630602 | 0.467811 | 4 |
| cross_experiment | experiment3_to_experiment2 | ECG_EEG_Pupil | 0.540000 | 0.452381 | 0.638750 | 0.599952 | 0.390000 | 5 |
| cross_experiment | experiment3_to_experiment2 | ECG | 0.535000 | 0.533589 | 0.496100 | 0.494133 | 0.419643 | 6 |
| cross_experiment | experiment3_to_experiment2 | ECG_Pupil | 0.515000 | 0.514405 | 0.553500 | 0.547690 | 0.399432 | 7 |
| within_experiment | experiment2 | ECG_EEG | 0.680000 | 0.679872 | 0.669000 | 0.629288 | 0.612727 | 1 |
| within_experiment | experiment2 | EEG | 0.635000 | 0.632344 | 0.714400 | 0.699261 | 0.546822 | 2 |
| within_experiment | experiment2 | ECG_EEG_Pupil | 0.630000 | 0.629407 | 0.723450 | 0.705881 | 0.545205 | 3 |
| within_experiment | experiment2 | EEG_Pupil | 0.610000 | 0.609844 | 0.703500 | 0.703919 | 0.501401 | 4 |
| within_experiment | experiment2 | Pupil | 0.565000 | 0.563680 | 0.645300 | 0.609993 | 0.469079 | 5 |
| within_experiment | experiment2 | ECG | 0.525000 | 0.499671 | 0.521600 | 0.532360 | 0.383333 | 6 |
| within_experiment | experiment2 | ECG_Pupil | 0.515000 | 0.501426 | 0.599100 | 0.590677 | 0.395810 | 7 |
| within_experiment | experiment3 | Pupil | 0.632895 | 0.626747 | 0.713844 | 0.737614 | 0.571174 | 1 |
| within_experiment | experiment3 | EEG | 0.614016 | 0.613103 | 0.672082 | 0.638909 | 0.523503 | 2 |
| within_experiment | experiment3 | ECG_Pupil | 0.595881 | 0.586187 | 0.659153 | 0.650985 | 0.516406 | 3 |
| within_experiment | experiment3 | EEG_Pupil | 0.583124 | 0.582876 | 0.687872 | 0.713009 | 0.489956 | 4 |
| within_experiment | experiment3 | ECG_EEG_Pupil | 0.582437 | 0.582302 | 0.694050 | 0.695866 | 0.463705 | 5 |
| within_experiment | experiment3 | ECG_EEG | 0.565103 | 0.560766 | 0.642906 | 0.614881 | 0.450338 | 6 |
| within_experiment | experiment3 | ECG | 0.452002 | 0.439132 | 0.469222 | 0.486685 | 0.338369 | 7 |

## Ranking policy

The displayed rank is descriptive only: pooled balanced accuracy, then pooled macro-F1, then pooled ROC-AUC, then alphabetical path. It did not affect model fitting, SHAP selection, validation, threshold selection, or the locked temporal protocol.

## Integrity coverage

- 28 report groups;
- 387 matched recordings across all 36 participants;
- 140,000 participant-bootstrap rows;
- zero numerical mismatches;
- zero report-hash mismatches;
- no post-hoc model refitting or threshold revision;
- no replacement of the primary four-second analysis.
