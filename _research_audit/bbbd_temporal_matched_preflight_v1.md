# BBBD Matched Temporal No-Fitting Preflight V1

Date: 2026-08-04

## Verdict

The complete matched temporal-sensitivity execution contract passed structural preflight before any predictive performance was observed.

## Coverage

- five durations: 4, 8, 16, 24, and 32 seconds;
- seven modality paths per duration;
- 36 within-experiment participant folds per duration;
- two source-only cross-experiment directions per duration;
- 1,330 duration/path/split contracts checked;
- 7,980 planned candidate-model fits;
- 1,330 planned SHAP-selector fits;
- 1,330 planned selected-candidate final refits;
- 10,640 total planned estimator fits.

## Verified safeguards

- Participant, recording, and segment roles are disjoint.
- Every split forms a complete participant-contained partition.
- Training-only retained-feature sets were verified.
- Total removed-feature accounting is exact.
- Constant/duplicate subtype counts are reported only when exposed by the locked production cleanup interface.
- Unexposed removal subtypes are explicitly recorded as unclassified; they are never guessed or invented.
- Cleanup-derived candidate counts match every locked execution plan.
- No SHAP selector or classifier was fitted.
- No threshold was selected.
- No validation, test, or target performance was observed.
- The original four-second analysis on 392 recordings remains primary.
- The matched four-second analysis is a sensitivity reference only.

## Path-level cleanup ranges

| Window | Path | Contracts | Original | Retained | Removed | Known constants | Known duplicates | Unclassified subtype | Candidates |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 s | EEG | 38 | 232232 | 232232 | 00 | 00 | 00 | 00 | 88 |
| 4 s | ECG | 38 | 1616 | 1616 | 00 | 00 | 00 | 00 | 22 |
| 4 s | Pupil | 38 | 2323 | 2121 | 22 | 00 | 00 | 22 | 44 |
| 4 s | ECG_EEG | 38 | 248248 | 248248 | 00 | 00 | 00 | 00 | 88 |
| 4 s | ECG_Pupil | 38 | 3939 | 3737 | 22 | 00 | 00 | 22 | 44 |
| 4 s | EEG_Pupil | 38 | 255255 | 253253 | 22 | 00 | 00 | 22 | 88 |
| 4 s | ECG_EEG_Pupil | 38 | 271271 | 269269 | 22 | 00 | 00 | 22 | 88 |
| 8 s | EEG | 38 | 232232 | 232232 | 00 | 00 | 00 | 00 | 88 |
| 8 s | ECG | 38 | 1616 | 1616 | 00 | 00 | 00 | 00 | 22 |
| 8 s | Pupil | 38 | 2323 | 2121 | 22 | 00 | 00 | 22 | 44 |
| 8 s | ECG_EEG | 38 | 248248 | 248248 | 00 | 00 | 00 | 00 | 88 |
| 8 s | ECG_Pupil | 38 | 3939 | 3737 | 22 | 00 | 00 | 22 | 44 |
| 8 s | EEG_Pupil | 38 | 255255 | 253253 | 22 | 00 | 00 | 22 | 88 |
| 8 s | ECG_EEG_Pupil | 38 | 271271 | 269269 | 22 | 00 | 00 | 22 | 88 |
| 16 s | EEG | 38 | 232232 | 232232 | 00 | 00 | 00 | 00 | 88 |
| 16 s | ECG | 38 | 1616 | 1616 | 00 | 00 | 00 | 00 | 22 |
| 16 s | Pupil | 38 | 2323 | 2121 | 22 | 00 | 00 | 22 | 44 |
| 16 s | ECG_EEG | 38 | 248248 | 248248 | 00 | 00 | 00 | 00 | 88 |
| 16 s | ECG_Pupil | 38 | 3939 | 3737 | 22 | 00 | 00 | 22 | 44 |
| 16 s | EEG_Pupil | 38 | 255255 | 253253 | 22 | 00 | 00 | 22 | 88 |
| 16 s | ECG_EEG_Pupil | 38 | 271271 | 269269 | 22 | 00 | 00 | 22 | 88 |
| 24 s | EEG | 38 | 232232 | 232232 | 00 | 00 | 00 | 00 | 88 |
| 24 s | ECG | 38 | 1616 | 1616 | 00 | 00 | 00 | 00 | 22 |
| 24 s | Pupil | 38 | 2323 | 2121 | 22 | 00 | 00 | 22 | 44 |
| 24 s | ECG_EEG | 38 | 248248 | 248248 | 00 | 00 | 00 | 00 | 88 |
| 24 s | ECG_Pupil | 38 | 3939 | 3737 | 22 | 00 | 00 | 22 | 44 |
| 24 s | EEG_Pupil | 38 | 255255 | 253253 | 22 | 00 | 00 | 22 | 88 |
| 24 s | ECG_EEG_Pupil | 38 | 271271 | 269269 | 22 | 00 | 00 | 22 | 88 |
| 32 s | EEG | 38 | 232232 | 232232 | 00 | 00 | 00 | 00 | 88 |
| 32 s | ECG | 38 | 1616 | 1616 | 00 | 00 | 00 | 00 | 22 |
| 32 s | Pupil | 38 | 2323 | 2121 | 22 | 00 | 00 | 22 | 44 |
| 32 s | ECG_EEG | 38 | 248248 | 248248 | 00 | 00 | 00 | 00 | 88 |
| 32 s | ECG_Pupil | 38 | 3939 | 3737 | 22 | 00 | 00 | 22 | 44 |
| 32 s | EEG_Pupil | 38 | 255255 | 253253 | 22 | 00 | 00 | 22 | 88 |
| 32 s | ECG_EEG_Pupil | 38 | 271271 | 269269 | 22 | 00 | 00 | 22 | 88 |

## Execution status

The evaluation runner may now be launched under the five locked duration-specific plans. Every duration, modality path, within-experiment setting, and transfer direction must be reported.
