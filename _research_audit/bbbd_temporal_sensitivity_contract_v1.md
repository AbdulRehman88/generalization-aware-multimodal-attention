# BBBD Temporal-Sensitivity Contract V1

Date: 2026-08-04

## Status

This contract was fixed after completion of the four-second primary BBBD
analysis but before extraction, model fitting, or performance inspection for
any longer BBBD window.

The analysis is therefore a post-primary sensitivity study. It cannot replace
the locked four-second primary result.

## Scientific question

How does raw physiological observation duration affect:

- attention-condition classification reliability;
- decision latency;
- modality-specific stability;
- multimodal fusion efficiency;
- cross-experiment robustness?

## Locked operating points

| Duration | Overlap | Stride | Role |
|---:|---:|---:|---|
| 4 s | 50% | 2 s | Primary real-time reference |
| 8 s | 50% | 4 s | Sensitivity |
| 16 s | 50% | 8 s | Sensitivity |
| 24 s | 50% | 12 s | Sensitivity |
| 32 s | 50% | 16 s | Sensitivity |

The 8-, 16-, 24-, and 32-second durations match the verified internal
long-window grid.

## Feature-generation rule

Features must be recomputed from the raw physiological recordings for every
duration.

Existing four-second feature vectors must not be averaged, concatenated, or
treated as equivalent to features extracted from a true longer raw signal
interval.

No window may cross a recording, task, condition, or session boundary.

## Locked cohort and modalities

The same complete-case cohort is retained:

- 36 participants;
- 392 recordings;
- 20 Experiment 2 participants;
- 16 Experiment 3 participants.

All seven modality paths are retained and must use aligned accepted windows.

## Evaluation contract

The sensitivity study preserves:

- nested participant-disjoint within-experiment evaluation;
- bidirectional source-only cross-experiment transfer;
- training-only cleanup and SHAP selection;
- validation-only model and threshold selection;
- recording-equal training weights;
- recording-level probability aggregation;
- 5,000 participant-cluster bootstrap repetitions.

## Initial exclusions

The first sensitivity analysis must not add:

- subject calibration;
- few-shot adaptation;
- personalized prototypes;
- probability-history smoothing;
- new models or model grids;
- raw 60-second or 120-second windows.

These factors answer different questions and would confound the isolated
effect of raw window duration.

## Reporting rule

Every duration, modality path, and evaluation direction must be reported.

The four-second result remains the primary real-time operating point even if a
longer window obtains higher performance.

The principal scientific output is the complete accuracy-versus-decision-time
profile, not only the maximum score.
