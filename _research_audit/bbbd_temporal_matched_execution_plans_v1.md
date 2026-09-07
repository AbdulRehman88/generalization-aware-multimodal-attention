# BBBD Matched Temporal Execution Plans V1

Date: 2026-08-04

## Verdict

Five duration-specific execution plans were constructed without feature selection, model fitting, threshold tuning, or performance inspection.

Every plan uses the same 387 recordings, all 36 participants, the same seven modality paths, and identical participant role assignments.

## Locked plans

| Window | Role | Outer folds | Transfer directions | Path/split evaluations | Planned estimator fits |
|---:|---|---:|---:|---:|---:|
| 4 s | matched_sensitivity_reference | 36 | 2 | 266 | 2128 |
| 8 s | post_primary_temporal_sensitivity | 36 | 2 | 266 | 2128 |
| 16 s | post_primary_temporal_sensitivity | 36 | 2 | 266 | 2128 |
| 24 s | post_primary_temporal_sensitivity | 36 | 2 | 266 | 2128 |
| 32 s | post_primary_temporal_sensitivity | 36 | 2 | 266 | 2128 |

## Aggregate planned workload

- 180 within-experiment outer folds;
- 10 cross-experiment directions;
- 1,330 path/split evaluations;
- 7,980 candidate-model fits;
- 1,330 SHAP-selector fits;
- 1,330 selected-candidate final refits;
- 10,640 total estimator fits.

## Safeguards

- Participant role assignments are identical across all durations.
- Candidate model and top-k specifications are identical.
- The matched four-second plan is a sensitivity reference only.
- The original four-second result on 392 recordings remains primary.
- No performance output existed when these plans were created.

The next permissible step is a no-fitting structural preflight over all 1,330 path/split contracts.
