# BBBD Seven-Path Feature Smoke V2

Date: 2026-08-02

## Locked cohort

- Cohort: participant-complete multimodal
- Participants: 36
- Recordings: 392
- Experiment 2 participants: 20
- Experiment 3 participants: 16
- Distracted recordings: 196
- Attentive recordings: 196

The cohort was fixed before feature selection or classification using only
prespecified ECG and pupil signal-usability requirements.

## Smoke design

The smoke extraction covered:

- both BBBD experiments;
- attentive and distracted conditions;
- three experiment-namespaced participants;
- six recordings;
- derivative EEG, uniformly processed raw ECG, and uniformly processed raw
  pupil signals.

Windowing used four-second windows with 50% overlap and two-second edge
trimming.

## Smoke result

- Candidate windows: 568
- Accepted windows: 565
- Rejected pupil windows: 3
- Rejected nonfinite-feature windows: 0
- Distracted windows: 281
- Attentive windows: 284

The three-window difference was caused by prespecified pupil-quality rejection.
No resampling or artificial balancing was performed.

## Feature dimensions

- EEG: 232
- ECG: 16
- Pupil: 23
- ECG+EEG: 248
- ECG+Pupil: 39
- EEG+Pupil: 255
- ECG+EEG+Pupil: 271

All seven tables had identical metadata alignment and finite numerical values.

## Scientific status

The seven-path feature-construction implementation passed its smoke test.

No classifier, SHAP ranking, feature selection, or performance evaluation was
performed. The complete 392-record cohort still requires extraction and
validation.

## Known limitation

The attentive and distracted conditions are confounded with session order,
repeat exposure, and the backward-counting dual task. Results must therefore
be framed as classification of the experimentally defined
attentive-versus-distracted condition.
