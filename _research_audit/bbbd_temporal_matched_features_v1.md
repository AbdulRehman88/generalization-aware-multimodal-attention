# BBBD Matched Temporal Feature Audit V1

Date: 2026-08-04

## Verdict

Five duration-specific matched feature roots were materialized successfully.

Each root contains the same 387 recordings from all 36 participants.

The operation filtered already extracted feature tables using the locked recording registry. It did not recompute raw physiological features, fit SHAP selectors, train classifiers, tune thresholds, or inspect predictive performance.

## Matched feature cohorts

| Window | Participants | Recordings | Rows per path | Pupil rejections | Nonfinite rejections |
|---:|---:|---:|---:|---:|---:|
| 4 s | 36 | 387 | 51511 | 1197 | 0 |
| 8 s | 36 | 387 | 25521 | 552 | 0 |
| 16 s | 36 | 387 | 12481 | 291 | 0 |
| 24 s | 36 | 387 | 8128 | 216 | 0 |
| 32 s | 36 | 387 | 5949 | 173 | 0 |

## Structural guarantees

- Every duration contains exactly the same 387 recordings.
- All 36 participants and both experimental conditions remain.
- All seven modality paths are aligned by segment identifier.
- Feature schemas exactly match their immutable source tables.
- Every retained recording has at least one accepted window.
- The original source feature files are byte-identical before and after materialization.

## Analysis boundary

The original four-second analysis on 392 recordings remains the primary real-time result.

The matched four-second root is only the fair sensitivity reference for comparison with 8, 16, 24, and 32 seconds.

The next permissible step is to create and structurally validate five execution plans using these matched feature roots.
