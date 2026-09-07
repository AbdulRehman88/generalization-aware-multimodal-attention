# BBBD Temporal-Sensitivity Implementation Audit V1

Date: 2026-08-04

## Verdict

The temporal-sensitivity implementation passed its raw-feature smoke audit.

The implementation is a thin wrapper over the existing locked BBBD raw-signal extractor. It does not introduce alternative preprocessing, feature definitions, SHAP selection, classifier fitting, or performance evaluation.

## Verified smoke grid

| Window | Overlap | Candidate windows | Accepted windows | Nonfinite rejections |
|---:|---:|---:|---:|---:|
| 8 s | 50% | 280 | 280 | 0 |
| 16 s | 50% | 136 | 136 | 0 |
| 24 s | 50% | 88 | 88 | 0 |
| 32 s | 50% | 64 | 64 | 0 |

## Structural guarantees

- All four durations completed the six-record smoke cohort.
- All seven modality feature files were produced.
- Every feature schema exactly matched the locked four-second schema.
- No duplicate segment identifiers were found.
- No nonfinite feature window was accepted.
- Candidate and accepted window counts decreased monotonically as duration increased.
- The locked four-second primary feature directory had identical tree hashes before and after extraction.

## Scientific status

No accuracy, balanced accuracy, macro-F1, ROC-AUC, PR-AUC, SHAP ranking, model selection, or classifier output has been generated for the longer-window sensitivity study.

The next permissible operation is full raw-feature extraction for the same four locked durations, followed by structural validation before any model is fitted.
