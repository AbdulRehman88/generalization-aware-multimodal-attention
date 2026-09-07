# BBBD Windowing Provenance Final V1

Date: 2026-08-03

## Decision

The locked primary BBBD evaluation uses:

- four-second windows;
- 50 percent overlap;
- two-second stride;
- two-second edge trimming.

This setting was present in the BBBD-specific configuration throughout its
committed history and remained unchanged through the pre-performance execution
plan and preflight. It was therefore selected before BBBD performance was
observed.

## BBBD-specific configuration history

- `6e145ed` (2026-08-02T16:17:05+09:00): 4.0-second windows, 50% overlap, 2.0-second edge trim  features: add BBBD complete-case seven-path extractor
- `17ab03d` (2026-08-02T18:35:58+09:00): 4.0-second windows, 50% overlap, 2.0-second edge trim  evaluation: lock BBBD participant-disjoint protocol
- `0a2d206` (2026-08-03T12:51:27+09:00): 4.0-second windows, 50% overlap, 2.0-second edge trim  evaluation: lock BBBD execution plan

Every committed version of `configs/bbbd.yaml` used the same four-second,
50-percent-overlap contract.

## Relationship to the legacy global configuration

No explicit numeric zero-overlap entry was detected in the current legacy global configuration.

Detected legacy overlap entries:

- No explicit overlap-valued field was detected.

The legacy global configuration is retained as historical project provenance.
It was not the configuration used to generate the locked BBBD features,
execution plan, predictions, or independently verified results.

## Post-performance non-overlap analysis

A new four-second, zero-overlap run may be conducted only as a separately
identified sensitivity analysis examining adjacent-window redundancy.

It must not:

- replace the locked primary BBBD analysis;
- be selected as primary based on its performance;
- trigger post-hoc changes to the model grid, feature-selection procedure,
  threshold rule, cohort, or claim framing;
- be used to portray whichever overlap setting performs better as
  prespecified.

It must preserve the same participant-complete cohort, seven modality paths,
nested participant-disjoint evaluation, source-only transfer protocol,
recording-level aggregation, and participant-cluster uncertainty analysis.

## Manuscript reporting policy

The manuscript must explicitly state the 50-percent overlap and two-second
stride.

Recordings remain the primary evaluation unit. Participants remain the
resampling and uncertainty unit. The 51,831 overlapping windows must not be
described as 51,831 independent observations.

Adjacent-window dependence should be acknowledged as a limitation even though:

- every participant is confined to one evaluation role;
- window probabilities are aggregated by recording;
- model fitting uses recording-equal weights;
- uncertainty intervals resample participants as clusters.

These safeguards prevent participant leakage and reduce recording-length
dominance, but they do not make adjacent overlapping windows statistically
independent.

## Evidence status

- Execution plan locked before performance: passed
- No classifier fitted during preflight: passed
- Completed path/split evaluations: 266
- Fold-aware result groups: 28
- Independent numerical verification: passed
- Windowing provenance audit: passed
