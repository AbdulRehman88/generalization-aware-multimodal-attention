# BBBD Primary Seven-Path Feature Cohort

Date: 2026-08-02
Source commit: `6e145ed`

## Cohort

- Cohort identity: participant-complete multimodal
- Participants: 36
- Recordings: 392
- Experiment 2: 20 participants and 200 recordings
- Experiment 3: 16 participants and 192 recordings
- Distracted recordings: 196
- Attentive recordings: 196

The cohort was fixed before feature selection or classification using only
prespecified signal-usability rules.

## Feature extraction

- Sampling frequency: 128 Hz
- Window duration: 4 seconds
- Window overlap: 50%
- Edge trim: 2 seconds
- Candidate windows: 53,512
- Accepted windows: 51,831
- Retention rate: 96.858649%
- Pupil-quality rejections: 1,681
- Nonfinite-feature rejections: 0

The accepted window counts are:

- Distracted: 25,436
- Attentive: 26,395

This window-level difference is retained as an observed consequence of
recording duration and prespecified pupil-quality filtering. It must not be
artificially balanced using test labels.

## Feature dimensions

- EEG: 232
- ECG: 16
- Pupil: 23
- ECG+EEG: 248
- ECG+Pupil: 39
- EEG+Pupil: 255
- ECG+EEG+Pupil: 271

All seven paths contain exactly the same 51,831 accepted windows in identical
metadata order.

## Required evaluation safeguards

1. No random window-level train/test split is permitted.
2. Every participant and recording must remain wholly inside one evaluation role.
3. Cleanup, scaling, SHAP ranking, model selection, and threshold selection
   must use training participants only.
4. Recording-level aggregation is the primary decision unit.
5. Participant-level metrics and participant-resampled confidence intervals
   are required.
6. Longer recordings must not receive disproportionate evaluation weight.
7. Direct modality comparisons must use the identical locked cohort and window
   registry.
8. No test-label-dependent window balancing is permitted.

## Claim boundary

BBBD supports classification of the experimentally defined
attentive-versus-distracted viewing condition. Condition is perfectly
confounded with session order, repeat exposure, and the backward-counting
dual task; therefore, the results cannot isolate attention as the sole causal
source of physiological differences.

## Status

Feature extraction is complete and verified. No classifier, feature selection,
SHAP ranking, or performance evaluation has yet been performed.

Combined artifact SHA-256:

`1d92873139817cd7f5291892d1dfaf7df226b75fe575c6b49abc6974ff1e2f25`
