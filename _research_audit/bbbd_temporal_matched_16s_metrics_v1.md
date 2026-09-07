# BBBD Matched Sixteen-Second Numerical Evidence V1

Date: 2026-08-05

## Verdict

All numerical values in the 28 matched sixteen-second reports were independently recomputed from recording-level predictions.

The audit found zero mismatches across pooled metrics, participant-level metrics, participant-macro summaries, validation-selected threshold decisions, 5,000-repetition participant-cluster confidence intervals, and report registry and manifest hashes.

## Scientific role

Sixteen seconds is a prespecified post-primary temporal-sensitivity duration. It does not replace the immutable primary four-second analysis on 392 recordings or the matched four-second sensitivity reference on 387 recordings.

These BBBD results measure attentive-versus-distracted experimental-condition classification. They do not isolate attention from session order, repeat exposure, fatigue, day/device drift, or the backward-counting dual task.

## Complete recording-level results

| Analysis | Setting | Path | Balanced accuracy | Macro-F1 | ROC-AUC | PR-AUC | Participant macro-F1 | Descriptive rank |
|---|---|---|---:|---:|---:|---:|---:|---:|
| cross_experiment | experiment2_to_experiment3 | EEG_Pupil | 0.660870 | 0.655375 | 0.699714 | 0.641166 | 0.570721 | 1 |
| cross_experiment | experiment2_to_experiment3 | EEG | 0.650686 | 0.647528 | 0.645080 | 0.628325 | 0.553055 | 2 |
| cross_experiment | experiment2_to_experiment3 | Pupil | 0.606922 | 0.594526 | 0.662586 | 0.655288 | 0.529651 | 3 |
| cross_experiment | experiment2_to_experiment3 | ECG_EEG_Pupil | 0.603204 | 0.570641 | 0.678375 | 0.696896 | 0.481820 | 4 |
| cross_experiment | experiment2_to_experiment3 | ECG_EEG | 0.595538 | 0.588392 | 0.648627 | 0.656224 | 0.511808 | 5 |
| cross_experiment | experiment2_to_experiment3 | ECG_Pupil | 0.585870 | 0.569481 | 0.611670 | 0.649379 | 0.452697 | 6 |
| cross_experiment | experiment2_to_experiment3 | ECG | 0.508238 | 0.508007 | 0.500686 | 0.526017 | 0.377463 | 7 |
| cross_experiment | experiment3_to_experiment2 | EEG | 0.650000 | 0.649439 | 0.658200 | 0.635543 | 0.570077 | 1 |
| cross_experiment | experiment3_to_experiment2 | EEG_Pupil | 0.590000 | 0.585859 | 0.632900 | 0.620242 | 0.488210 | 2 |
| cross_experiment | experiment3_to_experiment2 | ECG_EEG | 0.580000 | 0.571603 | 0.633500 | 0.572963 | 0.463654 | 3 |
| cross_experiment | experiment3_to_experiment2 | ECG_EEG_Pupil | 0.575000 | 0.572596 | 0.627500 | 0.598729 | 0.456794 | 4 |
| cross_experiment | experiment3_to_experiment2 | ECG | 0.550000 | 0.549955 | 0.500000 | 0.507513 | 0.439501 | 5 |
| cross_experiment | experiment3_to_experiment2 | Pupil | 0.535000 | 0.432494 | 0.664900 | 0.664058 | 0.394551 | 6 |
| cross_experiment | experiment3_to_experiment2 | ECG_Pupil | 0.535000 | 0.534056 | 0.567500 | 0.558801 | 0.431799 | 7 |
| within_experiment | experiment2 | ECG_EEG | 0.685000 | 0.684803 | 0.665150 | 0.678522 | 0.620677 | 1 |
| within_experiment | experiment2 | EEG | 0.620000 | 0.616897 | 0.666200 | 0.646422 | 0.524620 | 2 |
| within_experiment | experiment2 | ECG_EEG_Pupil | 0.600000 | 0.600000 | 0.659800 | 0.678517 | 0.500087 | 3 |
| within_experiment | experiment2 | EEG_Pupil | 0.580000 | 0.579622 | 0.644500 | 0.628396 | 0.475962 | 4 |
| within_experiment | experiment2 | Pupil | 0.580000 | 0.577295 | 0.644000 | 0.601642 | 0.487979 | 5 |
| within_experiment | experiment2 | ECG_Pupil | 0.535000 | 0.529816 | 0.611400 | 0.649096 | 0.418043 | 6 |
| within_experiment | experiment2 | ECG | 0.505000 | 0.461239 | 0.516700 | 0.571835 | 0.365485 | 7 |
| within_experiment | experiment3 | Pupil | 0.675515 | 0.670746 | 0.718879 | 0.750540 | 0.601801 | 1 |
| within_experiment | experiment3 | EEG_Pupil | 0.599600 | 0.598517 | 0.716705 | 0.696759 | 0.495839 | 2 |
| within_experiment | experiment3 | ECG_Pupil | 0.594680 | 0.592172 | 0.669451 | 0.652175 | 0.483457 | 3 |
| within_experiment | experiment3 | ECG_EEG_Pupil | 0.592963 | 0.592639 | 0.682838 | 0.699192 | 0.492264 | 4 |
| within_experiment | experiment3 | ECG_EEG | 0.583467 | 0.582589 | 0.685355 | 0.701191 | 0.473743 | 5 |
| within_experiment | experiment3 | EEG | 0.529119 | 0.529075 | 0.648741 | 0.646341 | 0.407001 | 6 |
| within_experiment | experiment3 | ECG | 0.484096 | 0.472124 | 0.486728 | 0.497756 | 0.338880 | 7 |

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
