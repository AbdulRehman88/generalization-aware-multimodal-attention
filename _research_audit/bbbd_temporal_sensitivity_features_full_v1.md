# Full BBBD Temporal-Sensitivity Feature Audit V1

Date: 2026-08-04

## Verdict

Full raw-feature extraction completed for all 392 recordings at 8, 16, 24, and 32 seconds.

No SHAP selection, classifier fitting, threshold tuning, or performance evaluation was performed.

## Recording availability

| Window | Processed | With accepted windows | Common matched cohort | Candidate windows | Accepted windows | Pupil rejections |
|---:|---:|---:|---:|---:|---:|---:|
| 4 s | 392 | 392 | 387 | 53512 | 51831 | 1681 |
| 8 s | 392 | 392 | 387 | 26472 | 25628 | 844 |
| 16 s | 392 | 392 | 387 | 12968 | 12500 | 468 |
| 24 s | 392 | 391 | 387 | 8472 | 8133 | 339 |
| 32 s | 392 | 387 | 387 | 6216 | 5949 | 267 |

## Objective common-recording cohort

The intersection across 4, 8, 16, 24, and 32 seconds contains 387 recordings from all 36 participants.

- Experiment 2: 200 recordings;
- Experiment 3: 187 recordings;
- distracted condition: 192 recordings;
- attentive condition: 195 recordings.

Five Experiment 3 recordings are excluded only from the matched temporal-sensitivity comparison because they have zero accepted aligned seven-path windows at 24 or 32 seconds.

No recording was selected or excluded using classifier performance.

## Primary versus sensitivity analysis

The original four-second evaluation on 392 recordings remains the immutable primary real-time analysis.

A separate four-second evaluation on the 387-recording intersection will serve only as the matched sensitivity reference for comparison with 8, 16, 24, and 32 seconds.

## Preserved safeguards

- The pupil-quality rule is unchanged.
- The 36-participant cohort is unchanged.
- Both experiments and both labels remain represented.
- All seven modality paths use identical recording membership.
- Participant-disjoint splitting remains mandatory.
- SHAP selection and tuning remain training-only.
- Every duration must be reported.
