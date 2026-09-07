# BBBD Matched Twenty-Four-Second Numerical Evidence V1

Date: 2026-08-06

## Verdict

All numerical values in the 28 matched twenty-four-second reports were independently recomputed from recording-level predictions.

The audit found zero mismatches across pooled metrics, participant-level metrics, participant-macro summaries, validation-selected threshold decisions, 5,000-repetition participant-cluster confidence intervals, and report registry and manifest hashes.

## Scientific role

Twenty-four seconds is a prespecified post-primary temporal-sensitivity duration. It does not replace the immutable primary four-second analysis on 392 recordings or the matched four-second sensitivity reference on 387 recordings.

These BBBD results measure attentive-versus-distracted experimental-condition classification. They do not isolate attention from session order, repeat exposure, fatigue, day/device drift, or the backward-counting dual task.

## Complete recording-level results

| Analysis | Setting | Path | Balanced accuracy | Macro-F1 | ROC-AUC | PR-AUC | Participant macro-F1 | Descriptive rank |
|---|---|---|---:|---:|---:|---:|---:|---:|
| cross_experiment | experiment2_to_experiment3 | EEG_Pupil | 0.665275 | 0.653372 | 0.680892 | 0.626253 | 0.575108 | 1 |
| cross_experiment | experiment2_to_experiment3 | EEG | 0.634897 | 0.632571 | 0.637757 | 0.627713 | 0.536461 | 2 |
| cross_experiment | experiment2_to_experiment3 | ECG_EEG_Pupil | 0.634725 | 0.631719 | 0.719680 | 0.739154 | 0.539431 | 3 |
| cross_experiment | experiment2_to_experiment3 | Pupil | 0.575172 | 0.558442 | 0.647941 | 0.630299 | 0.480916 | 4 |
| cross_experiment | experiment2_to_experiment3 | ECG_EEG | 0.565961 | 0.565054 | 0.631236 | 0.652746 | 0.457148 | 5 |
| cross_experiment | experiment2_to_experiment3 | ECG_Pupil | 0.553432 | 0.539733 | 0.614989 | 0.661137 | 0.419310 | 6 |
| cross_experiment | experiment2_to_experiment3 | ECG | 0.506865 | 0.504835 | 0.492677 | 0.538274 | 0.377647 | 7 |
| cross_experiment | experiment3_to_experiment2 | Pupil | 0.620000 | 0.619848 | 0.654500 | 0.652835 | 0.541639 | 1 |
| cross_experiment | experiment3_to_experiment2 | ECG_EEG_Pupil | 0.565000 | 0.556925 | 0.628900 | 0.618805 | 0.440260 | 2 |
| cross_experiment | experiment3_to_experiment2 | EEG_Pupil | 0.560000 | 0.557166 | 0.657300 | 0.699684 | 0.455659 | 3 |
| cross_experiment | experiment3_to_experiment2 | EEG | 0.560000 | 0.530667 | 0.631450 | 0.606799 | 0.434432 | 4 |
| cross_experiment | experiment3_to_experiment2 | ECG | 0.550000 | 0.547101 | 0.490600 | 0.483179 | 0.445613 | 5 |
| cross_experiment | experiment3_to_experiment2 | ECG_Pupil | 0.545000 | 0.518404 | 0.554600 | 0.550926 | 0.422569 | 6 |
| cross_experiment | experiment3_to_experiment2 | ECG_EEG | 0.490000 | 0.345399 | 0.707300 | 0.627895 | 0.338062 | 7 |
| within_experiment | experiment2 | EEG | 0.630000 | 0.629073 | 0.698950 | 0.694716 | 0.537660 | 1 |
| within_experiment | experiment2 | ECG_EEG | 0.615000 | 0.614913 | 0.673900 | 0.662130 | 0.527567 | 2 |
| within_experiment | experiment2 | ECG_EEG_Pupil | 0.580000 | 0.578947 | 0.681800 | 0.701584 | 0.487935 | 3 |
| within_experiment | experiment2 | Pupil | 0.570000 | 0.563718 | 0.647600 | 0.618566 | 0.476569 | 4 |
| within_experiment | experiment2 | EEG_Pupil | 0.560000 | 0.559824 | 0.690550 | 0.654458 | 0.427994 | 5 |
| within_experiment | experiment2 | ECG_Pupil | 0.535000 | 0.527619 | 0.605750 | 0.615818 | 0.424560 | 6 |
| within_experiment | experiment2 | ECG | 0.490000 | 0.418803 | 0.513500 | 0.521184 | 0.345444 | 7 |
| within_experiment | experiment3 | Pupil | 0.648341 | 0.645345 | 0.717506 | 0.748207 | 0.562107 | 1 |
| within_experiment | experiment3 | ECG_EEG_Pupil | 0.620309 | 0.620277 | 0.721110 | 0.717254 | 0.538887 | 2 |
| within_experiment | experiment3 | ECG_Pupil | 0.595023 | 0.590951 | 0.671167 | 0.649279 | 0.497284 | 3 |
| within_experiment | experiment3 | ECG_EEG | 0.581407 | 0.578537 | 0.684554 | 0.669948 | 0.482071 | 4 |
| within_experiment | experiment3 | EEG_Pupil | 0.571911 | 0.571886 | 0.680378 | 0.694652 | 0.473959 | 5 |
| within_experiment | experiment3 | EEG | 0.522826 | 0.520554 | 0.626659 | 0.629996 | 0.397551 | 6 |
| within_experiment | experiment3 | ECG | 0.462529 | 0.448052 | 0.466476 | 0.476699 | 0.336911 | 7 |

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
