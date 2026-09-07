
"""Complete no-fit real-data binding audit for deep-learning v1.3."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.data_loaders.deep_learning_real_data_binding import (
    ACTIVE_REGISTRY_V3,
    ADAPTIVE_DURATIONS,
    BBBD_CACHE_ROOT,
    PROTOCOL,
    REGISTRY,
    adaptive_temporal_roles,
    bind_bbbd_row,
    bind_internal_conventional_row,
    bind_internal_participant_split,
    load_internal_cache,
    path_requires_pupil,
    regenerate_frozen_conventional_index_map,
    scan_bbbd_cache_metadata,
    validate_internal_manifest_cache_alignment,
)
from src.models.deep_learning_training_runtime import (
    expand_execution_registry,
)


ROOT = Path(__file__).resolve().parents[2]

AUDIT_JSON = (
    ROOT
    / "_research_audit/"
      "deep_learning_real_data_binding_audit_v1.json"
)

AUDIT_MD = (
    ROOT
    / "_research_audit/"
      "deep_learning_real_data_binding_audit_v1.md"
)

BINDING_TABLE = (
    ROOT
    / "_research_audit/"
      "deep_learning_real_data_binding_summary_v1.csv"
)


def main():

    for path in [
        AUDIT_JSON,
        AUDIT_MD,
        BINDING_TABLE,
    ]:
        if path.exists():
            raise FileExistsError(
                path
            )


    protocol = yaml.safe_load(
        PROTOCOL.read_text(
            encoding="utf-8"
        )
    )

    if protocol[
        "protocol_revision"
    ] != "1.4":
        raise RuntimeError(
            "Expected protocol revision 1.3."
        )

    if protocol[
        "execution_gate"
    ][
        "real_training_allowed_now"
    ]:
        raise RuntimeError(
            "Real-training gate should still be closed during binding audit."
        )


    start_time = time.time()


    print(
        "===== A. INTERNAL MANIFEST / CACHE ALIGNMENT ====="
    )

    alignment = (
        validate_internal_manifest_cache_alignment()
    )

    print(
        alignment
    )


    print(
        "\n===== B. REPLAY FROZEN CONVENTIONAL SPLITS ====="
    )

    index_map, replay = (
        regenerate_frozen_conventional_index_map()
    )

    print(
        "Frozen registry replay rows:",
        len(
            replay
        ),
    )

    print(
        "Recovered unique conventional index hashes:",
        len(
            index_map
        ),
    )

    print(
        "Frozen registry replay parity: PASSED"
    )


    registry = pd.read_csv(
        ACTIVE_REGISTRY_V3,
        low_memory=False,
    )

    operations = expand_execution_registry(
        registry
    )


    summary_rows = []


    # ============================================================
    # C. INTERNAL CONVENTIONAL
    # ============================================================

    print(
        "\n===== C. INTERNAL CONVENTIONAL BINDINGS ====="
    )


    conventional = registry.loc[
        registry[
            "stage"
        ].astype(str)
        == "internal_conventional"
    ].copy()


    conventional_unique = (
        conventional[
            [
                "protocol",
                "split_id",
                "model",
                "path",
                "train_index_hash",
                "validation_index_hash",
                "test_index_hash",
            ]
        ]
        .drop_duplicates()
        .reset_index(
            drop=True
        )
    )


    for row in conventional_unique.to_dict(
        orient="records"
    ):

        binding = bind_internal_conventional_row(
            row,
            index_map,
        )

        summary_rows.append(
            {
                "stage":
                    "internal_conventional",

                "protocol":
                    row[
                        "protocol"
                    ],

                "split_id":
                    row[
                        "split_id"
                    ],

                "model":
                    row[
                        "model"
                    ],

                "path":
                    row[
                        "path"
                    ],

                "duration_seconds":
                    4,

                "calibration_budget_seconds_per_class":
                    "",

                "train_windows":
                    len(
                        binding[
                            "train_indices"
                        ]
                    ),

                "validation_windows":
                    len(
                        binding[
                            "validation_indices"
                        ]
                    ),

                "test_windows":
                    len(
                        binding[
                            "test_indices"
                        ]
                    ),

                "bindable":
                    True,
            }
        )


    print(
        "Unique conventional split/path bindings:",
        len(
            conventional_unique
        ),
    )

    print(
        "Conventional index/hash parity: PASSED"
    )


    # ============================================================
    # D. STRICT NESTED LOSO
    # ============================================================

    print(
        "\n===== D. INTERNAL STRICT NESTED LOSO BINDINGS ====="
    )


    nested = operations.loc[
        operations[
            "stage"
        ].astype(str)
        == "internal_nested_loso"
    ].copy()


    nested_unique = (
        nested[
            [
                "stage",
                "split_id",
                "model",
                "path",
                "duration_candidate_seconds",
                "train_participants",
                "validation_participants",
                "test_participants",
            ]
        ]
        .drop_duplicates()
        .reset_index(
            drop=True
        )
    )


    nested_pupil_minimums = {
        "train":
            None,

        "validation":
            None,

        "test":
            None,
    }


    for row in nested_unique.to_dict(
        orient="records"
    ):

        duration = int(
            row[
                "duration_candidate_seconds"
            ]
        )

        binding = bind_internal_participant_split(
            row,
            duration_seconds=duration,
        )


        counts = {
            "train":
                len(
                    binding[
                        "train_indices"
                    ]
                ),

            "validation":
                len(
                    binding[
                        "validation_indices"
                    ]
                ),

            "test":
                len(
                    binding[
                        "test_indices"
                    ]
                ),
        }


        if path_requires_pupil(
            row[
                "path"
            ]
        ):

            for key, value in counts.items():

                current = nested_pupil_minimums[
                    key
                ]

                nested_pupil_minimums[
                    key
                ] = (
                    value
                    if current is None
                    else min(
                        current,
                        value,
                    )
                )


        summary_rows.append(
            {
                "stage":
                    "internal_nested_loso",

                "protocol":
                    "strict_nested_loso",

                "split_id":
                    row[
                        "split_id"
                    ],

                "model":
                    row[
                        "model"
                    ],

                "path":
                    row[
                        "path"
                    ],

                "duration_seconds":
                    duration,

                "calibration_budget_seconds_per_class":
                    "",

                "train_windows":
                    counts[
                        "train"
                    ],

                "validation_windows":
                    counts[
                        "validation"
                    ],

                "test_windows":
                    counts[
                        "test"
                    ],

                "bindable":
                    True,
            }
        )


    print(
        "Unique nested split/path/duration bindings:",
        len(
            nested_unique
        ),
    )

    print(
        "Minimum Pupil-path train/validation/test windows:",
        nested_pupil_minimums,
    )

    print(
        "Nested LOSO class coverage + participant separation: PASSED"
    )


    # ============================================================
    # E. SUBJECT-ADAPTIVE ALL-CANDIDATE BINDINGS
    # ============================================================

    print(
        "\n===== E. SUBJECT-ADAPTIVE 30/60/120-S BINDINGS ====="
    )


    adaptive = operations.loc[
        operations[
            "stage"
        ].astype(str)
        == "internal_subject_adaptive"
    ].copy()


    adaptive_unique = (
        adaptive[
            [
                "split_id",
                "model",
                "path",
                "calibration_budget_seconds_per_class",
                "test_participants",
            ]
        ]
        .drop_duplicates()
        .reset_index(
            drop=True
        )
    )


    adaptive_candidate_checks = 0

    minimum_adaptive_calibration = None
    minimum_adaptive_test = None

    pupil_reduced_calibration_cases = 0


    for row in adaptive_unique.to_dict(
        orient="records"
    ):

        budget = int(
            float(
                row[
                    "calibration_budget_seconds_per_class"
                ]
            )
        )


        for duration in ADAPTIVE_DURATIONS:

            roles = adaptive_temporal_roles(
                row=row,
                duration_seconds=duration,
            )

            adaptive_candidate_checks += 1


            calibration_total = len(
                roles[
                    "calibration_indices"
                ]
            )

            test_total = len(
                roles[
                    "test_indices"
                ]
            )


            minimum_adaptive_calibration = (
                calibration_total
                if minimum_adaptive_calibration is None
                else min(
                    minimum_adaptive_calibration,
                    calibration_total,
                )
            )

            minimum_adaptive_test = (
                test_total
                if minimum_adaptive_test is None
                else min(
                    minimum_adaptive_test,
                    test_total,
                )
            )


            expected_raw_total = (
                roles[
                    "raw_calibration_windows_per_class"
                ]
                * 3
            )

            if calibration_total < expected_raw_total:

                if not path_requires_pupil(
                    row[
                        "path"
                    ]
                ):
                    raise RuntimeError(
                        "Non-Pupil adaptive path unexpectedly lost "
                        "calibration windows."
                    )

                pupil_reduced_calibration_cases += 1


            summary_rows.append(
                {
                    "stage":
                        "internal_subject_adaptive",

                    "protocol":
                        (
                            f"subject_adaptive_"
                            f"{budget}s_per_class"
                        ),

                    "split_id":
                        row[
                            "split_id"
                        ],

                    "model":
                        row[
                            "model"
                        ],

                    "path":
                        row[
                            "path"
                        ],

                    "duration_seconds":
                        duration,

                    "calibration_budget_seconds_per_class":
                        budget,

                    "train_windows":
                        "",

                    "validation_windows":
                        calibration_total,

                    "test_windows":
                        test_total,

                    "bindable":
                        True,
                }
            )


    expected_adaptive_checks = (
        13
        * 4
        * 3
        * 4
    )

    if adaptive_candidate_checks != expected_adaptive_checks:
        raise RuntimeError(
            f"Expected {expected_adaptive_checks} adaptive candidate "
            f"bindings, found {adaptive_candidate_checks}."
        )


    print(
        "Adaptive candidate bindings checked:",
        adaptive_candidate_checks,
    )

    print(
        "Minimum eligible adaptive calibration windows:",
        minimum_adaptive_calibration,
    )

    print(
        "Minimum eligible adaptive test windows:",
        minimum_adaptive_test,
    )

    print(
        "Pupil-path cases with fewer usable calibration windows "
        "inside the fixed temporal budget:",
        pupil_reduced_calibration_cases,
    )

    print(
        "Adaptive temporal boundaries + class coverage: PASSED"
    )


    # ============================================================
    # F. BBBD
    # ============================================================

    print(
        "\n===== F. BBBD WITHIN / CROSS-EXPERIMENT BINDINGS ====="
    )


    bbbd_metadata = scan_bbbd_cache_metadata()


    bbbd = registry.loc[
        registry[
            "stage"
        ].astype(str).isin(
            [
                "bbbd_within_experiment",
                "bbbd_cross_experiment",
            ]
        )
    ].copy()


    bbbd_unique = (
        bbbd[
            [
                "stage",
                "protocol",
                "dataset",
                "split_id",
                "model",
                "path",
                "train_participants",
                "validation_participants",
                "test_participants",
            ]
        ]
        .drop_duplicates()
        .reset_index(
            drop=True
        )
    )


    for row in bbbd_unique.to_dict(
        orient="records"
    ):

        binding = bind_bbbd_row(
            row,
            bbbd_metadata,
        )


        summary_rows.append(
            {
                "stage":
                    row[
                        "stage"
                    ],

                "protocol":
                    row[
                        "protocol"
                    ],

                "split_id":
                    row[
                        "split_id"
                    ],

                "model":
                    row[
                        "model"
                    ],

                "path":
                    row[
                        "path"
                    ],

                "duration_seconds":
                    4,

                "calibration_budget_seconds_per_class":
                    "",

                "train_windows":
                    binding[
                        "train"
                    ][
                        "windows"
                    ],

                "validation_windows":
                    binding[
                        "validation"
                    ][
                        "windows"
                    ],

                "test_windows":
                    binding[
                        "test"
                    ][
                        "windows"
                    ],

                "bindable":
                    True,
            }
        )


    print(
        "Unique BBBD split/path bindings:",
        len(
            bbbd_unique
        ),
    )

    print(
        "BBBD participants:",
        bbbd_metadata[
            "participant"
        ].nunique(),
    )

    print(
        "BBBD recordings:",
        len(
            bbbd_metadata
        ),
    )

    print(
        "BBBD accepted windows:",
        int(
            bbbd_metadata[
                "windows"
            ].sum()
        ),
    )

    print(
        "BBBD participant separation + binary class coverage: PASSED"
    )


    # ============================================================
    # G. GLOBAL OPERATION CONTRACT
    # ============================================================

    print(
        "\n===== G. GLOBAL OPERATION CONTRACT ====="
    )


    if len(
        operations
    ) != 3936:
        raise RuntimeError(
            "Expanded operation count is no longer 3,936."
        )

    if operations[
        "operation_id"
    ].nunique() != 3936:
        raise RuntimeError(
            "Operation IDs are no longer globally unique."
        )


    model_path_pairs = (
        operations[
            [
                "model",
                "path",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            [
                "model",
                "path",
            ]
        )
    )


    invalid_shallow = model_path_pairs.loc[
        (
            model_path_pairs[
                "model"
            ]
            == "ShallowConvNet"
        )
        & (
            model_path_pairs[
                "path"
            ]
            != "EEG"
        )
    ]

    if not invalid_shallow.empty:
        raise RuntimeError(
            "ShallowConvNet appears on a non-EEG path."
        )


    if len(
        model_path_pairs
    ) != 8:
        raise RuntimeError(
            f"Expected 8 model/path combinations, "
            f"found {len(model_path_pairs)}."
        )


    print(
        "Expanded operations: 3,936 / 3,936"
    )

    print(
        "Unique operation IDs: 3,936 / 3,936"
    )

    print(
        "Model/path combinations: 8"
    )


    # ============================================================
    # H. WRITE AUDIT
    # ============================================================

    summary = pd.DataFrame(
        summary_rows
    )

    summary.to_csv(
        BINDING_TABLE,
        index=False,
    )


    elapsed = (
        time.time()
        - start_time
    )


    report = {
        "identity":
            "deep_learning_real_data_binding_audit_v1",

        "protocol_revision":
            "1.4",

        "real_research_data_loaded":
            True,

        "model_initialized":
            False,

        "optimizer_created":
            False,

        "normalization_fitted":
            False,

        "loss_computed":
            False,

        "predictions_generated":
            False,

        "neural_network_fitting_performed":
            False,

        "performance_metrics_computed":
            False,

        "performance_observed":
            False,

        "internal_manifest_cache_alignment":
            alignment,

        "frozen_registry_replay_rows":
            int(
                len(
                    replay
                )
            ),

        "frozen_registry_replay_exact":
            True,

        "recovered_conventional_index_hashes":
            int(
                len(
                    index_map
                )
            ),

        "internal_conventional_unique_bindings":
            int(
                len(
                    conventional_unique
                )
            ),

        "internal_nested_unique_bindings":
            int(
                len(
                    nested_unique
                )
            ),

        "internal_nested_pupil_minimum_windows":
            {
                key:
                    int(
                        value
                    )
                for key, value
                in nested_pupil_minimums.items()
            },

        "adaptive_candidate_bindings_checked":
            adaptive_candidate_checks,

        "adaptive_minimum_eligible_calibration_windows":
            int(
                minimum_adaptive_calibration
            ),

        "adaptive_minimum_eligible_test_windows":
            int(
                minimum_adaptive_test
            ),

        "adaptive_pupil_reduced_calibration_cases":
            int(
                pupil_reduced_calibration_cases
            ),

        "adaptive_boundary_policy":
            (
                "Fixed temporal calibration and guard boundaries are "
                "never shifted to compensate for unavailable Pupil "
                "windows. Pupil paths use only quality-eligible windows "
                "inside those already-frozen temporal partitions."
            ),

        "bbbd_participants":
            int(
                bbbd_metadata[
                    "participant"
                ].nunique()
            ),

        "bbbd_recordings":
            int(
                len(
                    bbbd_metadata
                )
            ),

        "bbbd_accepted_windows":
            int(
                bbbd_metadata[
                    "windows"
                ].sum()
            ),

        "bbbd_unique_bindings":
            int(
                len(
                    bbbd_unique
                )
            ),

        "expanded_fit_adaptation_operations":
            int(
                len(
                    operations
                )
            ),

        "unique_operation_ids":
            int(
                operations[
                    "operation_id"
                ].nunique()
            ),

        "model_path_combinations":
            int(
                len(
                    model_path_pairs
                )
            ),

        "binding_summary_rows":
            int(
                len(
                    summary
                )
            ),

        "all_real_data_bindings_passed":
            True,

        "elapsed_seconds":
            elapsed,

        "real_training_gate_open":
            False,

        "next_action":
            (
                "implement the final registry-driven launcher using "
                "these exact binding functions, open the execution "
                "gate, and start full real-data training"
            ),
    }


    AUDIT_JSON.write_text(
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


    AUDIT_MD.write_text(
        f"""# Deep-Learning Real-Data Binding Audit v1

## Status

**PASSED  all frozen real-data bindings are executable without fitting.**

## Internal data

- Manifest/cache positional alignment: passed.
- Base windows: {alignment['rows']:,}.
- Participants: {alignment['participants']}.
- Pupil-valid base windows: {alignment['pupil_valid']:,}.
- Pupil-invalid base windows: {alignment['pupil_invalid']:,}.

The exact conventional split index arrays were recovered by replaying
the already-frozen execution-plan generator in an isolated temporary
directory and intercepting its original `hash_indices()` calls. The
replayed 3,000-row registry matched frozen registry v2 exactly.

## Strict nested LOSO

All participant-disjoint train/validation/test bindings were verified
for all candidate durations (8, 16, 24, and 32 s) and all eight
model/path combinations.

Minimum Pupil-path window counts across checked bindings:

- training: {nested_pupil_minimums['train']:,}
- validation: {nested_pupil_minimums['validation']:,}
- outer test: {nested_pupil_minimums['test']:,}

All three numeric classes were present in every required partition.

## Subject-adaptive evaluation

Checked {adaptive_candidate_checks:,} pre-performance combinations
covering all 13 outer participants, eight model/path combinations,
30/60/120-s-per-class calibration budgets, and all four possible
validation-selected durations.

Fixed calibration and guard boundaries were not shifted to replace
unavailable Pupil windows. Pupil paths use only quality-eligible windows
that naturally fall inside the frozen temporal partitions.

Minimum eligible calibration windows across all bindings:
{minimum_adaptive_calibration:,}.

Minimum eligible test windows across all bindings:
{minimum_adaptive_test:,}.

Pupil-path candidate cases with fewer usable calibration windows than
the raw temporal budget contains:
{pupil_reduced_calibration_cases:,}.

## BBBD

- Participants: {bbbd_metadata['participant'].nunique()}.
- Recordings: {len(bbbd_metadata):,}.
- Accepted aligned windows: {int(bbbd_metadata['windows'].sum()):,}.
- All within-experiment and cross-experiment participant partitions
  are disjoint where required.
- Training, validation, and test partitions retain both binary classes.

## Global execution plan

- Expanded fit/adaptation operations: {len(operations):,}.
- Globally unique operation IDs: {operations['operation_id'].nunique():,}.
- Model/path combinations: {len(model_path_pairs)}.
- ShallowConvNet remains EEG-only.

## Performance firewall

- Model initialized: no.
- Optimizer created: no.
- Normalization fitted: no.
- Loss computed: no.
- Predictions generated: no.
- Neural-network fitting: no.
- Performance metrics computed: no.

The real-training gate remains closed until the final launcher is
committed.
""".rstrip()
        + "\n",
        encoding="utf-8",
    )


    print(
        "\n===== FINAL REAL-DATA BINDING VERDICT ====="
    )

    print(
        "All real-data bindings: PASSED"
    )

    print(
        "Conventional frozen-index replay: PASSED"
    )

    print(
        "Nested LOSO binding: PASSED"
    )

    print(
        "Adaptive 30/60/120-s all-duration binding: PASSED"
    )

    print(
        "BBBD within/cross binding: PASSED"
    )

    print(
        "Expanded operations: 3,936 / 3,936"
    )

    print(
        "Model initialized: False"
    )

    print(
        "Optimizer created: False"
    )

    print(
        "Loss computed: False"
    )

    print(
        "Neural-network fitting performed: False"
    )

    print(
        "Performance observed: False"
    )

    print(
        "Real-training gate open: False"
    )

    print(
        "Elapsed seconds:",
        round(
            elapsed,
            2,
        ),
    )

    print(
        "Created:",
        AUDIT_JSON.relative_to(
            ROOT
        ),
    )

    print(
        "Created:",
        AUDIT_MD.relative_to(
            ROOT
        ),
    )

    print(
        "Created:",
        BINDING_TABLE.relative_to(
            ROOT
        ),
    )


if __name__ == "__main__":
    main()
