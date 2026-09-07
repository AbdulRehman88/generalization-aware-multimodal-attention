# BBBD Matched Eight-Second Numerical Evidence V1

Date: 2026-08-05

## Verdict

All numerical values in the 28 matched eight-second reports were independently recomputed from recording-level predictions.

The audit found zero mismatches across pooled metrics, participant-level metrics, participant-macro summaries, validation-selected threshold decisions, 5,000-repetition participant-cluster confidence intervals, and report registry and manifest hashes.

## Scientific role

Eight seconds is a prespecified post-primary temporal-sensitivity duration. It does not replace the immutable primary four-second analysis on 392 recordings or the matched four-second sensitivity reference on 387 recordings.

These BBBD results measure attentive-versus-distracted experimental-condition classification. They do not isolate attention from session order, repeat exposure, fatigue, day/device drift, or the backward-counting dual task.

## Complete recording-level results

| Analysis | Setting | Path | Balanced accuracy | Macro-F1 | ROC-AUC | PR-AUC | Participant macro-F1 | Descriptive rank |
|---|---|---|---:|---:|---:|---:|---:|---:|
| cross_experiment | experiment2_to_experiment3 | EEG_Pupil | 0.688387 | 0.686604 | 0.708352 | 0.650981 | 0.608519 | 1 |
| cross_experiment | experiment2_to_experiment3 | Pupil | 0.652746 | 0.652367 | 0.673684 | 0.665359 | 0.584426 | 2 |
| cross_experiment | experiment2_to_experiment3 | ECG_EEG | 0.647941 | 0.620784 | 0.688101 | 0.646190 | 0.543454 | 3 |
| cross_experiment | experiment2_to_experiment3 | ECG_EEG_Pupil | 0.634382 | 0.629746 | 0.695767 | 0.702158 | 0.547332 | 4 |
| cross_experiment | experiment2_to_experiment3 | EEG | 0.629462 | 0.626747 | 0.647368 | 0.608118 | 0.535383 | 5 |
| cross_experiment | experiment2_to_experiment3 | ECG_Pupil | 0.574314 | 0.565621 | 0.622998 | 0.650339 | 0.440143 | 6 |
| cross_experiment | experiment2_to_experiment3 | ECG | 0.490217 | 0.484848 | 0.489245 | 0.519753 | 0.363628 | 7 |
| cross_experiment | experiment3_to_experiment2 | EEG | 0.610000 | 0.606815 | 0.635150 | 0.621546 | 0.524510 | 1 |
| cross_experiment | experiment3_to_experiment2 | ECG_EEG | 0.580000 | 0.578483 | 0.604300 | 0.570537 | 0.461676 | 2 |
| cross_experiment | experiment3_to_experiment2 | ECG_EEG_Pupil | 0.560000 | 0.560000 | 0.607300 | 0.589736 | 0.448685 | 3 |
| cross_experiment | experiment3_to_experiment2 | EEG_Pupil | 0.560000 | 0.502488 | 0.666500 | 0.651416 | 0.431149 | 4 |
| cross_experiment | experiment3_to_experiment2 | Pupil | 0.520000 | 0.398119 | 0.676850 | 0.674208 | 0.381863 | 5 |
| cross_experiment | experiment3_to_experiment2 | ECG_Pupil | 0.510000 | 0.507587 | 0.569700 | 0.559346 | 0.392971 | 6 |
| cross_experiment | experiment3_to_experiment2 | ECG | 0.500000 | 0.472073 | 0.498000 | 0.522114 | 0.369890 | 7 |
| within_experiment | experiment2 | ECG_EEG_Pupil | 0.640000 | 0.639423 | 0.681800 | 0.668502 | 0.563804 | 1 |
| within_experiment | experiment2 | EEG | 0.600000 | 0.594156 | 0.681500 | 0.646160 | 0.497192 | 2 |
| within_experiment | experiment2 | EEG_Pupil | 0.590000 | 0.589631 | 0.645500 | 0.625742 | 0.486610 | 3 |
| within_experiment | experiment2 | Pupil | 0.585000 | 0.584740 | 0.628300 | 0.590477 | 0.495105 | 4 |
| within_experiment | experiment2 | ECG_EEG | 0.575000 | 0.563106 | 0.614500 | 0.598655 | 0.465196 | 5 |
| within_experiment | experiment2 | ECG | 0.540000 | 0.511885 | 0.509900 | 0.518930 | 0.405745 | 6 |
| within_experiment | experiment2 | ECG_Pupil | 0.525000 | 0.509994 | 0.609950 | 0.580940 | 0.389261 | 7 |
| within_experiment | experiment3 | EEG_Pupil | 0.647140 | 0.647049 | 0.713616 | 0.720035 | 0.565928 | 1 |
| within_experiment | experiment3 | Pupil | 0.633410 | 0.623742 | 0.744622 | 0.784153 | 0.564053 | 2 |
| within_experiment | experiment3 | ECG_EEG_Pupil | 0.609439 | 0.609447 | 0.677346 | 0.680057 | 0.509452 | 3 |
| within_experiment | experiment3 | ECG_Pupil | 0.589931 | 0.584384 | 0.641991 | 0.618013 | 0.463692 | 4 |
| within_experiment | experiment3 | ECG_EEG | 0.548627 | 0.541238 | 0.620481 | 0.637430 | 0.418048 | 5 |
| within_experiment | experiment3 | EEG | 0.528776 | 0.528319 | 0.629233 | 0.613027 | 0.390665 | 6 |
| within_experiment | experiment3 | ECG | 0.484611 | 0.477289 | 0.476773 | 0.488640 | 0.344884 | 7 |

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
