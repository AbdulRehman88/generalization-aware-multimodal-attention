# BBBD Matched Four-Second Numerical Evidence V1

Date: 2026-08-04

## Verdict

All numerical values in the 28 matched four-second reports were independently recomputed from recording-level predictions.

The audit found zero mismatches across pooled metrics, participant-level metrics, participant-macro summaries, stored threshold decisions, 5,000-repetition participant-cluster confidence intervals, report registry hashes, and report manifest hashes.

## Scientific boundary

These results measure attentive-versus-distracted experimental-condition classification in BBBD. They do not isolate attention from session order, repeat exposure, fatigue, day/device drift, or the backward-counting dual task.

The matched four-second result is a temporal-sensitivity reference only. It does not replace the immutable primary four-second analysis on 392 recordings.

## Complete recording-level results

| Analysis | Setting | Path | Balanced accuracy | Macro-F1 | ROC-AUC | PR-AUC | Participant macro-F1 | Descriptive rank |
|---|---|---|---:|---:|---:|---:|---:|---:|
| cross_experiment | experiment2_to_experiment3 | ECG_EEG_Pupil | 0.645824 | 0.618635 | 0.667506 | 0.680814 | 0.552424 | 1 |
| cross_experiment | experiment2_to_experiment3 | EEG | 0.643021 | 0.621039 | 0.654805 | 0.631993 | 0.552153 | 2 |
| cross_experiment | experiment2_to_experiment3 | ECG_EEG | 0.631979 | 0.604602 | 0.672654 | 0.640927 | 0.526345 | 3 |
| cross_experiment | experiment2_to_experiment3 | EEG_Pupil | 0.597254 | 0.573059 | 0.694966 | 0.691390 | 0.503543 | 4 |
| cross_experiment | experiment2_to_experiment3 | Pupil | 0.567277 | 0.524898 | 0.704577 | 0.683849 | 0.455164 | 5 |
| cross_experiment | experiment2_to_experiment3 | ECG_Pupil | 0.556121 | 0.556099 | 0.569680 | 0.604132 | 0.427205 | 6 |
| cross_experiment | experiment2_to_experiment3 | ECG | 0.510984 | 0.492208 | 0.525858 | 0.537727 | 0.378019 | 7 |
| cross_experiment | experiment3_to_experiment2 | Pupil | 0.610000 | 0.596941 | 0.674900 | 0.664969 | 0.518205 | 1 |
| cross_experiment | experiment3_to_experiment2 | ECG_EEG_Pupil | 0.580000 | 0.576570 | 0.625700 | 0.595846 | 0.471612 | 2 |
| cross_experiment | experiment3_to_experiment2 | EEG_Pupil | 0.580000 | 0.556541 | 0.638800 | 0.631771 | 0.461428 | 3 |
| cross_experiment | experiment3_to_experiment2 | ECG_Pupil | 0.575000 | 0.552337 | 0.606200 | 0.633811 | 0.471336 | 4 |
| cross_experiment | experiment3_to_experiment2 | ECG_EEG | 0.520000 | 0.461340 | 0.665200 | 0.586267 | 0.369231 | 5 |
| cross_experiment | experiment3_to_experiment2 | EEG | 0.515000 | 0.419839 | 0.623100 | 0.620541 | 0.379464 | 6 |
| cross_experiment | experiment3_to_experiment2 | ECG | 0.490000 | 0.481233 | 0.493500 | 0.520007 | 0.362111 | 7 |
| within_experiment | experiment2 | ECG_EEG_Pupil | 0.600000 | 0.595960 | 0.679800 | 0.659321 | 0.481030 | 1 |
| within_experiment | experiment2 | Pupil | 0.595000 | 0.593771 | 0.655550 | 0.620874 | 0.502778 | 2 |
| within_experiment | experiment2 | EEG_Pupil | 0.590000 | 0.589631 | 0.652400 | 0.642454 | 0.475293 | 3 |
| within_experiment | experiment2 | EEG | 0.575000 | 0.574904 | 0.649600 | 0.648695 | 0.465631 | 4 |
| within_experiment | experiment2 | ECG_EEG | 0.550000 | 0.536608 | 0.552700 | 0.563260 | 0.423932 | 5 |
| within_experiment | experiment2 | ECG | 0.525000 | 0.489124 | 0.521450 | 0.552445 | 0.389194 | 6 |
| within_experiment | experiment2 | ECG_Pupil | 0.515000 | 0.501426 | 0.611900 | 0.596470 | 0.383516 | 7 |
| within_experiment | experiment3 | EEG_Pupil | 0.631693 | 0.630636 | 0.698970 | 0.698986 | 0.540552 | 1 |
| within_experiment | experiment3 | ECG_EEG_Pupil | 0.611156 | 0.606747 | 0.670824 | 0.675447 | 0.509795 | 2 |
| within_experiment | experiment3 | Pupil | 0.600286 | 0.596670 | 0.740503 | 0.777168 | 0.521310 | 3 |
| within_experiment | experiment3 | ECG_Pupil | 0.588730 | 0.588047 | 0.631236 | 0.606217 | 0.472516 | 4 |
| within_experiment | experiment3 | EEG | 0.565618 | 0.563651 | 0.604920 | 0.600295 | 0.449630 | 5 |
| within_experiment | experiment3 | ECG_EEG | 0.555606 | 0.555336 | 0.629062 | 0.635682 | 0.434272 | 6 |
| within_experiment | experiment3 | ECG | 0.504119 | 0.474847 | 0.481808 | 0.509990 | 0.359374 | 7 |

## Ranking policy

The displayed rank is descriptive only: pooled balanced accuracy, then pooled macro-F1, then pooled ROC-AUC, then alphabetical path name. It did not affect training, validation, threshold selection, or the prespecified evaluation protocol.

## Integrity coverage

- 28 report groups;
- 387 matched recordings across all 36 participants;
- 140,000 participant-bootstrap rows;
- zero numerical mismatches;
- zero report-hash mismatches;
- no post-hoc model refitting or threshold revision.
