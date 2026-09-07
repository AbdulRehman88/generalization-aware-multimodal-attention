# BBBD Matched Temporal-Sensitivity Evidence V1

Date: 2026-08-06

## Verdict

The complete prespecified 4-, 8-, 16-, 24-, and 32-second matched temporal analysis has been aggregated from independently audited duration-level evidence.

The combined evidence contains 140 setting-path-duration results representing 700,000 participant-cluster bootstrap rows and zero numerical mismatches.

The matched four-second analysis uses the common 387-recording sensitivity cohort and serves only as the temporal reference. It does not replace the immutable primary four-second analysis on 392 recordings.

## Best path at each duration

| Analysis | Setting | Duration | Best path | Balanced accuracy | Macro-F1 | ROC-AUC | PR-AUC |
|---|---|---:|---|---:|---:|---:|---:|
| cross_experiment | experiment2_to_experiment3 | 4 s | ECG_EEG_Pupil | 0.645824 | 0.618635 | 0.667506 | 0.680814 |
| cross_experiment | experiment2_to_experiment3 | 8 s | EEG_Pupil | 0.688387 | 0.686604 | 0.708352 | 0.650981 |
| cross_experiment | experiment2_to_experiment3 | 16 s | EEG_Pupil | 0.660870 | 0.655375 | 0.699714 | 0.641166 |
| cross_experiment | experiment2_to_experiment3 | 24 s | EEG_Pupil | 0.665275 | 0.653372 | 0.680892 | 0.626253 |
| cross_experiment | experiment2_to_experiment3 | 32 s | EEG_Pupil | 0.650515 | 0.646585 | 0.671281 | 0.640214 |
| cross_experiment | experiment3_to_experiment2 | 4 s | Pupil | 0.610000 | 0.596941 | 0.674900 | 0.664969 |
| cross_experiment | experiment3_to_experiment2 | 8 s | EEG | 0.610000 | 0.606815 | 0.635150 | 0.621546 |
| cross_experiment | experiment3_to_experiment2 | 16 s | EEG | 0.650000 | 0.649439 | 0.658200 | 0.635543 |
| cross_experiment | experiment3_to_experiment2 | 24 s | Pupil | 0.620000 | 0.619848 | 0.654500 | 0.652835 |
| cross_experiment | experiment3_to_experiment2 | 32 s | EEG_Pupil | 0.635000 | 0.634991 | 0.649600 | 0.668252 |
| within_experiment | experiment2 | 4 s | ECG_EEG_Pupil | 0.600000 | 0.595960 | 0.679800 | 0.659321 |
| within_experiment | experiment2 | 8 s | ECG_EEG_Pupil | 0.640000 | 0.639423 | 0.681800 | 0.668502 |
| within_experiment | experiment2 | 16 s | ECG_EEG | 0.685000 | 0.684803 | 0.665150 | 0.678522 |
| within_experiment | experiment2 | 24 s | EEG | 0.630000 | 0.629073 | 0.698950 | 0.694716 |
| within_experiment | experiment2 | 32 s | ECG_EEG | 0.680000 | 0.679872 | 0.669000 | 0.629288 |
| within_experiment | experiment3 | 4 s | EEG_Pupil | 0.631693 | 0.630636 | 0.698970 | 0.698986 |
| within_experiment | experiment3 | 8 s | EEG_Pupil | 0.647140 | 0.647049 | 0.713616 | 0.720035 |
| within_experiment | experiment3 | 16 s | Pupil | 0.675515 | 0.670746 | 0.718879 | 0.750540 |
| within_experiment | experiment3 | 24 s | Pupil | 0.648341 | 0.645345 | 0.717506 | 0.748207 |
| within_experiment | experiment3 | 32 s | Pupil | 0.632895 | 0.626747 | 0.713844 | 0.737614 |

## Descriptive best result across durations

| Analysis | Setting | Duration | Path | Balanced accuracy | Macro-F1 | Delta BA vs matched 4 s | Delta F1 vs matched 4 s |
|---|---|---:|---|---:|---:|---:|---:|
| cross_experiment | experiment2_to_experiment3 | 8 s | EEG_Pupil | 0.688387 | 0.686604 | +0.091133 | +0.113545 |
| cross_experiment | experiment3_to_experiment2 | 16 s | EEG | 0.650000 | 0.649439 | +0.135000 | +0.229600 |
| within_experiment | experiment2 | 16 s | ECG_EEG | 0.685000 | 0.684803 | +0.135000 | +0.148195 |
| within_experiment | experiment3 | 16 s | Pupil | 0.675515 | 0.670746 | +0.075229 | +0.074076 |

## Scientific interpretation boundary

The duration comparisons are descriptive. Every prespecified duration and modality path is retained. No duration was selected after observing performance, no model was refitted during aggregation, and no claim of statistically significant duration superiority or a minimum reliable decision time is made.

The results characterize attentive-versus-distracted experimental-condition classification on BBBD. They do not isolate attention from all effects of task structure, session order, repeat exposure, fatigue, or day/device variation.

## Integrity coverage

- five prespecified durations;
- seven modality paths;
- four evaluation settings;
- 140 complete setting-path-duration rows;
- 700,000 participant-bootstrap rows represented;
- zero numerical mismatches;
- no additional fitting or feature selection;
- no replacement of the primary four-second result.
