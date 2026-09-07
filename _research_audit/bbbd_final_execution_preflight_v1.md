# BBBD Final Execution Preflight V1

Date: 2026-08-03

## Status

This audit used the real locked BBBD feature tables to verify execution roles,
training-only cleanup, and effective candidate counts.

No feature selector or classifier was fitted. No validation, outer-test, or
cross-experiment performance was computed or observed.

## Scope

- Seven modality paths
- 36 within-experiment outer folds per path
- Two cross-experiment directions per path
- Path/split contracts checked: 266
- Within-experiment path/fold contracts: 252
- Cross-experiment path/direction contracts: 14

All training, validation, and test roles were verified as participant-,
recording-, and segment-disjoint and were matched against the locked execution
plan.

## Training-only feature cleanup

For every path and split, constant-feature and exact duplicate-feature removal
was evaluated using training participants only. Validation and test rows did not
influence cleanup.

Candidate-count mismatches: 0

## Locked computation budget

- Expected candidate model fits: 1596
- Observed candidate model fits after training-only cleanup:
  1596
- SHAP selector fits: 266
- Selected-candidate final refits: 266
- Expected total estimator fits: 2128
- Observed total estimator fits: 2128

## Decision

The full BBBD evaluation may begin only if:

- candidate-count mismatches equal zero;
- observed candidate fits equal the locked candidate-fit budget;
- observed total estimator fits equal the locked total-fit budget;
- all unit tests pass;
- the repository is clean.

No performance-based alteration of this execution plan is permitted after the
real evaluation begins.
