# BBBD Evaluation Runner Implementation V1

Date: 2026-08-03

## Status

The full evaluation runner has been implemented against the execution plan
locked at commit `0a2d206`.

No classifier has been fitted to BBBD features, and no BBBD outer-test or
cross-experiment performance has been observed.

## Leakage safeguards

The runner enforces:

- participant-disjoint training, validation, and test roles;
- recording and segment containment within one role;
- training-only constant and duplicate-feature removal;
- training-only selector fitting and SHAP ranking;
- recording-equal window weights during every fitting step;
- validation-only model-family and effective-top-k selection;
- validation-only recording-level threshold selection;
- one untouched outer participant per within-experiment fold;
- no target-experiment tuning in cross-experiment transfer.

## Evaluation outputs

Each completed path and split writes:

- training-only feature-cleanup decisions;
- SHAP feature ranking;
- all validation candidates;
- threshold-search results;
- selected candidate and selected features;
- test-window probabilities;
- recording-level probabilities and decisions;
- participant-specific metrics;
- aggregate metrics;
- an atomic completion record.

A completed split is resumable only when its stored split identity matches the
requested split. A nonempty incomplete directory is never overwritten
silently.

## Aggregation

The primary decision unit is the recording. Window probabilities are averaged
within each recording. Participant-specific recording metrics are retained for
participant-macro reporting and participant-cluster confidence intervals.

## Validation stage

Only synthetic grouped data may be used during the implementation smoke test.
The synthetic test must exercise one nested participant-disjoint split, one
source-only cross-experiment transfer split, model selection, SHAP selection,
threshold selection, output serialization, and resume behavior.

The real BBBD evaluation remains unstarted.
