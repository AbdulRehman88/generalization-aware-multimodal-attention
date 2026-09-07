# BBBD Fold-Aware Reporting V1

Date: 2026-08-03

## Reason for this lock

Within-experiment outer folds may select different recording-level decision
thresholds from their inner validation participants.

Applying one global threshold to all pooled out-of-fold predictions would
discard the locked fold-specific decision rule. Shifting probabilities to
simulate a common threshold would alter cross-fold probability ordering and
could distort ROC-AUC and PR-AUC.

## Locked metric policy

Threshold-dependent metrics use each recording's stored decision:

- accuracy
- balanced accuracy
- macro-F1

Probability-ranking metrics use the original, untouched probability:

- ROC-AUC
- PR-AUC

No threshold shifting or probability recalibration is performed during
reporting.

## Participant reporting and uncertainty

Participant-specific metrics are computed from that participant's recording
predictions. Participant-macro metrics average those participant-specific
values.

Confidence intervals resample participants as clusters. Every sampled
participant retains:

- all recordings;
- the original out-of-fold or transfer probability;
- the stored validation-selected threshold;
- the stored recording decision.

Because participant bootstrap samples with replacement, each sampled draw is
assigned a unique bootstrap participant identity. Copied recordings receive
draw-specific recording identities while retaining their original source
recording identifiers. This prevents identity collisions without treating
repeated draws as independent original participants.

## Validation status

The reporting implementation was tested only with synthetic data specifically
constructed so that:

- fold-specific thresholds yield correct decisions;
- a single global threshold yields an incorrect decision;
- the untouched probabilities preserve perfect class ranking.

No BBBD classifier was fitted and no BBBD performance was observed while
locking this reporting policy.
