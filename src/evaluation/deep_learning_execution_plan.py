
"""Frozen execution registry for deep-learning baselines.

This module constructs the complete pre-performance execution plan.
It does not instantiate, fit, score, or select a neural network.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    ShuffleSplit,
    StratifiedKFold,
    train_test_split,
)


ROOT = Path(__file__).resolve().parents[2]

PROTOCOL_PATH = (
    ROOT
    / "configs/deep_learning_baselines.yaml"
)

INTERNAL_MANIFEST = (
    ROOT
    / "outputs/revision/preprocessed/internal_xr/window_manifest.csv"
)

GROUPED_FOLDS = (
    ROOT
    / "outputs/revision/evaluation/internal_xr_pilot/fold_metrics.csv"
)

NESTED_ROLES = (
    ROOT
    / "outputs/revision/evaluation/"
      "internal_xr_loso_long_window/inner_validation_results.csv"
)

BBBD_EXECUTION_PLAN = (
    ROOT
    / "_research_audit/bbbd_evaluation_execution_plan_v1.json"
)

REGISTRY_PATH = (
    ROOT
    / "artifacts/revision/manifests/"
      "deep_learning_execution_registry_v2.csv"
)

AUDIT_JSON = (
    ROOT
    / "_research_audit/deep_learning_execution_plan_v2.json"
)

AUDIT_MD = (
    ROOT
    / "_research_audit/deep_learning_execution_plan_v2.md"
)


MODEL_PATHS = [
    (
        "ShallowConvNet",
        "EEG",
    ),
    (
        "MultibranchTCN",
        "EEG",
    ),
    (
        "MultibranchTCN",
        "ECG",
    ),
    (
        "MultibranchTCN",
        "Pupil",
    ),
    (
        "MultibranchTCN",
        "ECG_EEG",
    ),
    (
        "MultibranchTCN",
        "ECG_Pupil",
    ),
    (
        "MultibranchTCN",
        "EEG_Pupil",
    ),
    (
        "MultibranchTCN",
        "ECG_EEG_Pupil",
    ),
]


PUPIL_PATHS = {
    "Pupil",
    "ECG_Pupil",
    "EEG_Pupil",
    "ECG_EEG_Pupil",
}


def sha256_file(
    path: Path,
) -> str:

    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as stream:

        for block in iter(
            lambda:
                stream.read(
                    1024 * 1024
                ),
            b"",
        ):
            digest.update(
                block
            )

    return digest.hexdigest()


def hash_indices(
    values,
) -> str:

    array = np.asarray(
        values,
        dtype=np.int64,
    )

    array = np.sort(
        array
    )

    return hashlib.sha256(
        array.tobytes()
    ).hexdigest()


def split_semicolon(
    value,
) -> list[str]:

    if pd.isna(
        value
    ):
        return []

    return [
        item.strip()
        for item
        in str(
            value
        ).split(";")
        if item.strip()
    ]


def semicolon(
    values,
) -> str:

    return ";".join(
        str(value)
        for value
        in values
    )


def robust_bool(
    series: pd.Series,
) -> pd.Series:

    if series.dtype == bool:
        return series

    return (
        series
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(
            {
                "true",
                "1",
                "yes",
                "y",
            }
        )
    )


def assert_disjoint(
    *sets_to_check,
) -> None:

    converted = [
        set(values)
        for values
        in sets_to_check
    ]

    for left_index in range(
        len(converted)
    ):
        for right_index in range(
            left_index + 1,
            len(converted),
        ):
            overlap = (
                converted[left_index]
                & converted[right_index]
            )

            if overlap:
                raise RuntimeError(
                    "Split overlap detected: "
                    f"{sorted(overlap)[:20]}"
                )


def participant_validation_split(
    participants: list[str],
    *,
    seed: int,
    count: int = 2,
) -> tuple[list[str], list[str]]:

    values = sorted(
        set(
            participants
        )
    )

    if len(values) <= count:
        raise ValueError(
            "Not enough participants for grouped validation split."
        )

    rng = np.random.default_rng(
        seed
    )

    permutation = rng.permutation(
        len(values)
    )

    validation_indices = set(
        permutation[
            :count
        ].tolist()
    )

    validation = sorted(
        values[index]
        for index
        in validation_indices
    )

    training = sorted(
        value
        for index, value
        in enumerate(
            values
        )
        if index not in validation_indices
    )

    assert_disjoint(
        training,
        validation,
    )

    return (
        training,
        validation,
    )


def eligible_manifest(
    manifest: pd.DataFrame,
    path: str,
) -> pd.DataFrame:

    frame = manifest.copy()

    if path in PUPIL_PATHS:
        frame = frame.loc[
            robust_bool(
                frame[
                    "pupil_available"
                ]
            )
        ].copy()

    return frame


def make_row(
    *,
    stage: str,
    protocol: str,
    split_id: str,
    model: str,
    path: str,
    seed: int,
    dataset: str,
    train_participants="",
    validation_participants="",
    test_participants="",
    train_index_hash="",
    validation_index_hash="",
    test_index_hash="",
    candidate_durations="",
    calibration_budget_seconds_per_class="",
    dependency="",
    notes="",
):
    return {
        "stage":
            stage,

        "protocol":
            protocol,

        "dataset":
            dataset,

        "split_id":
            split_id,

        "model":
            model,

        "path":
            path,

        "seed":
            int(seed),

        "train_participants":
            train_participants,

        "validation_participants":
            validation_participants,

        "test_participants":
            test_participants,

        "train_index_hash":
            train_index_hash,

        "validation_index_hash":
            validation_index_hash,

        "test_index_hash":
            test_index_hash,

        "candidate_durations_seconds":
            candidate_durations,

        "calibration_budget_seconds_per_class":
            calibration_budget_seconds_per_class,

        "dependency":
            dependency,

        "notes":
            notes,
    }


def main():

    for path in [
        REGISTRY_PATH,
        AUDIT_JSON,
        AUDIT_MD,
    ]:
        if path.exists():
            raise FileExistsError(
                path
            )

    protocol = yaml.safe_load(
        PROTOCOL_PATH.read_text(
            encoding="utf-8"
        )
    )

    if protocol[
        "protocol_revision"
    ] != "1.2":
        raise RuntimeError(
            "Expected protocol revision 1.2."
        )

    seeds = [
        int(value)
        for value
        in protocol[
            "randomness"
        ][
            "training_seeds"
        ]
    ]

    if seeds != [
        42,
        3407,
        2026,
    ]:
        raise RuntimeError(
            f"Unexpected training seeds: {seeds}"
        )

    adaptive_budgets = [
        int(value)
        for value
        in protocol[
            "internal_subject_adaptive"
        ][
            "calibration_budgets_seconds_per_class"
        ]
    ]

    if adaptive_budgets != [
        30,
        60,
        120,
    ]:
        raise RuntimeError(
            f"Unexpected adaptive budgets: {adaptive_budgets}"
        )

    manifest = pd.read_csv(
        INTERNAL_MANIFEST,
        low_memory=False,
    )

    if len(
        manifest
    ) != 3315:
        raise RuntimeError(
            "Internal manifest row count mismatch."
        )

    manifest = manifest.reset_index(
        drop=True
    )

    grouped = pd.read_csv(
        GROUPED_FOLDS,
        low_memory=False,
    )

    nested = pd.read_csv(
        NESTED_ROLES,
        low_memory=False,
    )

    nested_columns = [
        "outer_fold",
        "outer_test_participant",
        "inner_training_participants",
        "inner_validation_participants",
    ]

    nested = (
        nested[
            nested_columns
        ]
        .drop_duplicates()
        .sort_values(
            "outer_fold"
        )
        .reset_index(
            drop=True
        )
    )

    if len(
        nested
    ) != 13:
        raise RuntimeError(
            "Expected 13 internal nested LOSO roles."
        )

    bbbd_plan = json.loads(
        BBBD_EXECUTION_PLAN.read_text(
            encoding="utf-8"
        )
    )

    within_bbbd = bbbd_plan[
        "within_experiment_folds"
    ]

    cross_bbbd = bbbd_plan[
        "cross_experiment_directions"
    ]

    if len(
        within_bbbd
    ) != 36:
        raise RuntimeError(
            "Expected 36 BBBD within-experiment folds."
        )

    if len(
        cross_bbbd
    ) != 2:
        raise RuntimeError(
            "Expected 2 BBBD cross-experiment directions."
        )

    rows = []


    # ============================================================
    # A. INTERNAL CONVENTIONAL  PARTICIPANT GROUPED 5-FOLD
    # ============================================================

    for _, fold_row in grouped.iterrows():

        fold = int(
            fold_row[
                "fold"
            ]
        )

        outer_training = split_semicolon(
            fold_row[
                "training_participants"
            ]
        )

        outer_test = split_semicolon(
            fold_row[
                "test_participants"
            ]
        )

        actual_train, validation = (
            participant_validation_split(
                outer_training,
                seed=42 + fold,
                count=2,
            )
        )

        assert_disjoint(
            actual_train,
            validation,
            outer_test,
        )

        for model, path in MODEL_PATHS:

            eligible = eligible_manifest(
                manifest,
                path,
            )

            train_indices = eligible.index[
                eligible[
                    "participant"
                ].isin(
                    actual_train
                )
            ].to_numpy()

            val_indices = eligible.index[
                eligible[
                    "participant"
                ].isin(
                    validation
                )
            ].to_numpy()

            test_indices = eligible.index[
                eligible[
                    "participant"
                ].isin(
                    outer_test
                )
            ].to_numpy()

            assert_disjoint(
                train_indices,
                val_indices,
                test_indices,
            )

            for seed in seeds:

                rows.append(
                    make_row(
                        stage=
                            "internal_conventional",

                        protocol=
                            "primary_grouped_5fold",

                        split_id=
                            f"fold_{fold:02d}",

                        model=
                            model,

                        path=
                            path,

                        seed=
                            seed,

                        dataset=
                            "internal_xr",

                        train_participants=
                            semicolon(
                                actual_train
                            ),

                        validation_participants=
                            semicolon(
                                validation
                            ),

                        test_participants=
                            semicolon(
                                outer_test
                            ),

                        train_index_hash=
                            hash_indices(
                                train_indices
                            ),

                        validation_index_hash=
                            hash_indices(
                                val_indices
                            ),

                        test_index_hash=
                            hash_indices(
                                test_indices
                            ),

                        notes=
                            (
                                "Participant-disjoint 5-fold conventional "
                                "evaluation; two training-fold participants "
                                "reserved deterministically for early stopping."
                            ),
                    )
                )


    # ============================================================
    # B. INTERNAL WINDOW-LEVEL LEGACY DIAGNOSTICS
    # ============================================================

    legacy_specs = [
        (
            "legacy_stratified_window_5fold",
            StratifiedKFold(
                n_splits=5,
                shuffle=True,
                random_state=42,
            ),
        ),
        (
            "legacy_repeated_stratified_window_5x3",
            RepeatedStratifiedKFold(
                n_splits=5,
                n_repeats=3,
                random_state=42,
            ),
        ),
        (
            "legacy_shuffle_split_window",
            ShuffleSplit(
                n_splits=10,
                test_size=0.2,
                random_state=42,
            ),
        ),
    ]

    for model, path in MODEL_PATHS:

        eligible = eligible_manifest(
            manifest,
            path,
        )

        global_indices = eligible.index.to_numpy(
            dtype=np.int64
        )

        labels = eligible[
            "label"
        ].to_numpy(
            dtype=int
        )

        dummy = np.zeros(
            (
                len(
                    eligible
                ),
                1,
            ),
            dtype=float,
        )

        for protocol_name, splitter in legacy_specs:

            outer_splits = list(
                splitter.split(
                    dummy,
                    labels,
                )
            )

            expected_split_count = {
                "legacy_stratified_window_5fold":
                    5,

                "legacy_repeated_stratified_window_5x3":
                    15,

                "legacy_shuffle_split_window":
                    10,
            }[
                protocol_name
            ]

            if len(
                outer_splits
            ) != expected_split_count:
                raise RuntimeError(
                    f"{protocol_name}: split-count mismatch."
                )

            for split_number, (
                outer_train_local,
                outer_test_local,
            ) in enumerate(
                outer_splits,
                start=1,
            ):

                outer_train_global = global_indices[
                    outer_train_local
                ]

                outer_test_global = global_indices[
                    outer_test_local
                ]

                outer_train_labels = manifest.loc[
                    outer_train_global,
                    "label",
                ].to_numpy(
                    dtype=int
                )

                train_global, val_global = (
                    train_test_split(
                        outer_train_global,
                        test_size=0.20,
                        random_state=
                            (
                                42000
                                + split_number
                                + (
                                    1000
                                    if protocol_name
                                    == "legacy_repeated_stratified_window_5x3"
                                    else 0
                                )
                                + (
                                    2000
                                    if protocol_name
                                    == "legacy_shuffle_split_window"
                                    else 0
                                )
                            ),
                        shuffle=True,
                        stratify=
                            outer_train_labels,
                    )
                )

                assert_disjoint(
                    train_global,
                    val_global,
                    outer_test_global,
                )

                for seed in seeds:

                    rows.append(
                        make_row(
                            stage=
                                "internal_conventional",

                            protocol=
                                protocol_name,

                            split_id=
                                f"split_{split_number:02d}",

                            model=
                                model,

                            path=
                                path,

                            seed=
                                seed,

                            dataset=
                                "internal_xr",

                            train_index_hash=
                                hash_indices(
                                    train_global
                                ),

                            validation_index_hash=
                                hash_indices(
                                    val_global
                                ),

                            test_index_hash=
                                hash_indices(
                                    outer_test_global
                                ),

                            notes=
                                (
                                    "Legacy in-distribution window-level "
                                    "diagnostic. Validation is 20% of the "
                                    "outer training partition, stratified "
                                    "and fixed before model fitting."
                                ),
                        )
                    )


    # ============================================================
    # C. INTERNAL PRIMARY STRICT NESTED LOSO
    # ============================================================

    durations = [
        8,
        16,
        24,
        32,
    ]

    duration_string = semicolon(
        durations
    )

    for _, role in nested.iterrows():

        outer_fold = int(
            role[
                "outer_fold"
            ]
        )

        outer_test = [
            str(
                role[
                    "outer_test_participant"
                ]
            )
        ]

        inner_train = split_semicolon(
            role[
                "inner_training_participants"
            ]
        )

        inner_val = split_semicolon(
            role[
                "inner_validation_participants"
            ]
        )

        assert_disjoint(
            inner_train,
            inner_val,
            outer_test,
        )

        if len(
            inner_train
        ) != 10:
            raise RuntimeError(
                f"Outer fold {outer_fold}: expected 10 inner-training participants."
            )

        if len(
            inner_val
        ) != 2:
            raise RuntimeError(
                f"Outer fold {outer_fold}: expected 2 validation participants."
            )

        for model, path in MODEL_PATHS:
            for seed in seeds:

                job_id = (
                    f"internal_nested_loso/"
                    f"fold_{outer_fold:02d}/"
                    f"{model}/{path}/seed_{seed}"
                )

                rows.append(
                    make_row(
                        stage=
                            "internal_nested_loso",

                        protocol=
                            "strict_nested_loso",

                        split_id=
                            f"outer_{outer_fold:02d}",

                        model=
                            model,

                        path=
                            path,

                        seed=
                            seed,

                        dataset=
                            "internal_xr",

                        train_participants=
                            semicolon(
                                inner_train
                            ),

                        validation_participants=
                            semicolon(
                                inner_val
                            ),

                        test_participants=
                            semicolon(
                                outer_test
                            ),

                        candidate_durations=
                            duration_string,

                        notes=
                            (
                                "Four duration candidates are trained "
                                "using inner-training participants. "
                                "Duration and early-stopping epoch are "
                                "selected using inner validation only. "
                                "The best validation checkpoint is applied "
                                "directly to the untouched outer participant; "
                                "no outer-test refit occurs."
                            ),
                    )
                )

                for budget_seconds_per_class in adaptive_budgets:

                    rows.append(
                        make_row(
                            stage=
                                "internal_subject_adaptive",

                            protocol=
                                (
                                    f"subject_adaptive_"
                                    f"{budget_seconds_per_class}s_per_class"
                                ),

                            split_id=
                                f"outer_{outer_fold:02d}",

                            model=
                                model,

                            path=
                                path,

                            seed=
                                seed,

                            dataset=
                                "internal_xr",

                            train_participants=
                                semicolon(
                                    inner_train
                                ),

                            validation_participants=
                                semicolon(
                                    inner_val
                                ),

                            test_participants=
                                semicolon(
                                    outer_test
                                ),

                            candidate_durations=
                                duration_string,

                            calibration_budget_seconds_per_class=
                                budget_seconds_per_class,

                            dependency=
                                job_id,

                            notes=
                                (
                                    "Reuse the duration and global checkpoint "
                                    "selected by the corresponding strict nested "
                                    "LOSO job. Adapt classifier head only using "
                                    f"{budget_seconds_per_class} labelled seconds "
                                    "per class from the outer participant, followed "
                                    "by one selected-window-duration guard before "
                                    "testing. All 30/60/120-s budgets are mandatory "
                                    "reporting conditions."
                                ),
                        )
                    )


    # ============================================================
    # D. BBBD WITHIN-EXPERIMENT NESTED LOSO
    # ============================================================

    for fold_number, fold in enumerate(
        within_bbbd,
        start=1,
    ):

        outer_test = [
            str(
                fold[
                    "outer_test_participant"
                ]
            )
        ]

        inner_train = [
            str(value)
            for value
            in fold[
                "inner_training_participants"
            ]
        ]

        inner_val = [
            str(value)
            for value
            in fold[
                "inner_validation_participants"
            ]
        ]

        assert_disjoint(
            inner_train,
            inner_val,
            outer_test,
        )

        experiment = (
            outer_test[0].split(
                "::",
                1,
            )[0]
        )

        for model, path in MODEL_PATHS:
            for seed in seeds:

                rows.append(
                    make_row(
                        stage=
                            "bbbd_within_experiment",

                        protocol=
                            "nested_loso",

                        split_id=
                            f"{experiment}_outer_{fold_number:02d}",

                        model=
                            model,

                        path=
                            path,

                        seed=
                            seed,

                        dataset=
                            experiment,

                        train_participants=
                            semicolon(
                                inner_train
                            ),

                        validation_participants=
                            semicolon(
                                inner_val
                            ),

                        test_participants=
                            semicolon(
                                outer_test
                            ),

                        notes=
                            (
                                "Locked BBBD 4-s/50%-overlap accepted-window "
                                "universe. Early stopping uses source "
                                "validation participants only. Best validation "
                                "checkpoint is evaluated directly on the "
                                "outer participant."
                            ),
                    )
                )


    # ============================================================
    # E. BBBD CROSS-EXPERIMENT SOURCE-ONLY TRANSFER
    # ============================================================

    for direction in cross_bbbd:

        name = str(
            direction[
                "name"
            ]
        )

        source_train = [
            str(value)
            for value
            in direction[
                "source_training_participants"
            ]
        ]

        source_validation = [
            str(value)
            for value
            in direction[
                "source_validation_participants"
            ]
        ]

        target_experiment = str(
            direction[
                "target"
            ]
        )

        # The locked BBBD execution-plan JSON stores only the
        # target participant COUNT in roles.target_test.participants.
        # Explicit target identities are therefore recovered from
        # the already-locked within-experiment LOSO outer-test
        # participant universe for the target experiment.
        target_test = sorted(
            {
                str(
                    fold[
                        "outer_test_participant"
                    ]
                )
                for fold
                in within_bbbd
                if str(
                    fold[
                        "outer_test_participant"
                    ]
                ).split(
                    "::",
                    1,
                )[0]
                == target_experiment
            }
        )

        declared_target_count = int(
            direction[
                "roles"
            ][
                "target_test"
            ][
                "participants"
            ]
        )

        if len(
            target_test
        ) != declared_target_count:
            raise RuntimeError(
                f"{name}: derived target participant count "
                f"{len(target_test)} does not match locked "
                f"count {declared_target_count}."
            )

        if not all(
            participant.startswith(
                target_experiment
                + "::"
            )
            for participant
            in target_test
        ):
            raise RuntimeError(
                f"{name}: target participant namespace mismatch."
            )

        assert_disjoint(
            source_train,
            source_validation,
            target_test,
        )

        for model, path in MODEL_PATHS:
            for seed in seeds:

                rows.append(
                    make_row(
                        stage=
                            "bbbd_cross_experiment",

                        protocol=
                            "source_only_transfer",

                        split_id=
                            name,

                        model=
                            model,

                        path=
                            path,

                        seed=
                            seed,

                        dataset=
                            "bbbd",

                        train_participants=
                            semicolon(
                                source_train
                            ),

                        validation_participants=
                            semicolon(
                                source_validation
                            ),

                        test_participants=
                            semicolon(
                                target_test
                            ),

                        notes=
                            (
                                "Model selection and early stopping use "
                                "source experiment only. No target participant "
                                "is used for normalization, model selection, "
                                "threshold selection, or early stopping."
                            ),
                    )
                )


    registry = pd.DataFrame(
        rows
    )

    registry = registry.sort_values(
        [
            "stage",
            "protocol",
            "split_id",
            "model",
            "path",
            "seed",
        ]
    ).reset_index(
        drop=True
    )


    # ============================================================
    # EXECUTION COUNTS
    # ============================================================

    stage_counts = (
        registry[
            "stage"
        ]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    expected_stage_counts = {
        "internal_conventional":
            840,

        "internal_nested_loso":
            312,

        "internal_subject_adaptive":
            936,

        "bbbd_within_experiment":
            864,

        "bbbd_cross_experiment":
            48,
    }

    if stage_counts != expected_stage_counts:
        raise RuntimeError(
            "Execution stage-count mismatch:\n"
            f"expected={expected_stage_counts}\n"
            f"actual={stage_counts}"
        )

    if len(
        registry
    ) != 3000:
        raise RuntimeError(
            f"Expected 3,000 registry jobs, found {len(registry)}."
        )

    nested_candidate_fits = (
        expected_stage_counts[
            "internal_nested_loso"
        ]
        * 4
    )

    planned_fit_operations = (
        expected_stage_counts[
            "internal_conventional"
        ]
        + nested_candidate_fits
        + expected_stage_counts[
            "internal_subject_adaptive"
        ]
        + expected_stage_counts[
            "bbbd_within_experiment"
        ]
        + expected_stage_counts[
            "bbbd_cross_experiment"
        ]
    )

    if planned_fit_operations != 3936:
        raise RuntimeError(
            "Expected 3,936 planned fit/adaptation operations."
        )


    # ============================================================
    # SAVE
    # ============================================================

    REGISTRY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    AUDIT_JSON.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    registry.to_csv(
        REGISTRY_PATH,
        index=False,
    )

    source_hashes = {
        str(
            path.relative_to(
                ROOT
            )
        ):
            sha256_file(
                path
            )
        for path in [
            PROTOCOL_PATH,
            INTERNAL_MANIFEST,
            GROUPED_FOLDS,
            NESTED_ROLES,
            BBBD_EXECUTION_PLAN,
        ]
    }

    report = {
        "identity":
            "deep_learning_execution_plan_v2",

        "protocol_revision":
            "1.2",

        "performance_blind":
            True,

        "real_model_training_performed":
            False,

        "performance_observed":
            False,

        "training_seeds":
            seeds,

        "model_path_combinations":
            [
                {
                    "model":
                        model,
                    "path":
                        path,
                }
                for model, path
                in MODEL_PATHS
            ],

        "model_path_combination_count":
            8,

        "registry_rows":
            len(
                registry
            ),

        "stage_counts":
            stage_counts,

        "internal_conventional_split_instances":
            {
                "primary_grouped_5fold":
                    5,

                "legacy_stratified_window_5fold":
                    5,

                "legacy_repeated_stratified_window_5x3":
                    15,

                "legacy_shuffle_split_window":
                    10,

                "total":
                    35,
            },

        "internal_nested_loso_outer_folds":
            13,

        "internal_nested_candidate_durations_seconds":
            durations,

        "internal_subject_adaptive_budgets_seconds_per_class":
            adaptive_budgets,

        "highlighted_subject_adaptive_budget_seconds_per_class":
            120,

        "bbbd_within_experiment_outer_folds":
            36,

        "bbbd_cross_experiment_directions":
            2,

        "planned_fit_operations":
            {
                "internal_conventional":
                    840,

                "internal_nested_duration_candidates":
                    nested_candidate_fits,

                "internal_subject_adaptations":
                    expected_stage_counts[
                        "internal_subject_adaptive"
                    ],

                "bbbd_within_experiment":
                    864,

                "bbbd_cross_experiment":
                    48,

                "total":
                    planned_fit_operations,
            },

        "validation_policy":
            {
                "grouped_conventional":
                    (
                        "Two participants are deterministically reserved "
                        "from each outer training fold using seed 42+fold."
                    ),

                "window_level_conventional":
                    (
                        "20% stratified validation subset is reserved "
                        "inside each outer training partition."
                    ),

                "nested_loso":
                    (
                        "Reuse the exact locked two-participant "
                        "inner-validation roles."
                    ),

                "bbbd":
                    (
                        "Reuse the exact locked source training and "
                        "validation participant roles."
                    ),
            },

        "checkpoint_policy":
            (
                "Use the checkpoint selected by validation balanced "
                "accuracy with macro-F1 tie-break. No refit using outer "
                "test data is permitted."
            ),

        "pupil_policy":
            {
                "internal":
                    "r only",

                "bbbd":
                    "pupil_size only",

                "model_channels":
                    1,
            },

        "runtime_progress_contract":
            protocol[
                "execution_reporting"
            ][
                "progress_bar"
            ],

        "source_hashes":
            source_hashes,

        "registry_sha256":
            sha256_file(
                REGISTRY_PATH
            ),

        "next_gate":
            "real-data forward-only dry-run",

        "real_training_allowed_after_plan_only":
            False,
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


    lines = [
        "# Deep-Learning Execution Plan v1",
        "",
        "## Status",
        "",
        "**Frozen before real neural-network fitting.**",
        "",
        f"- Registry jobs: {len(registry):,}.",
        f"- Model/path combinations: 8.",
        f"- Training seeds: {seeds}.",
        "",
        "## Registry composition",
        "",
        "| Stage | Jobs |",
        "|---|---:|",
    ]

    for stage, count in sorted(
        stage_counts.items()
    ):
        lines.append(
            f"| {stage} | {count:,} |"
        )

    lines.extend(
        [
            "",
            "## Planned fit/adaptation operations",
            "",
            f"- Internal conventional: 840.",
            f"- Internal nested LOSO duration candidates: {nested_candidate_fits:,}.",
            f"- Internal 30/60/120-s/class head adaptations: "
            f"{expected_stage_counts['internal_subject_adaptive']:,}.",
            f"- BBBD within-experiment: 864.",
            f"- BBBD cross-experiment: 48.",
            f"- Total planned operations: {planned_fit_operations:,}.",
            "",
            "## Leakage controls",
            "",
            "- Outer test data are excluded from normalization, early stopping, duration selection, and model selection.",
            "- Strict internal LOSO reuses the exact locked participant roles.",
            "- BBBD reuses the exact locked within-experiment and source-only transfer participant roles.",
            "- Window-level protocols remain explicitly labelled legacy in-distribution diagnostics.",
            "- The internal pupil branch uses only `r`; BBBD uses only `pupil_size`.",
            "- No best seed is selected; all three prespecified seeds are reported.",
            "- No outer-test refit is allowed.",
            "",
            "## Training status",
            "",
            "No neural-network fitting or performance evaluation was performed while generating this registry.",
        ]
    )

    AUDIT_MD.write_text(
        "\n".join(
            lines
        ).rstrip()
        + "\n",
        encoding="utf-8",
    )

    print(
        "Registry rows:",
        len(
            registry
        ),
    )

    print(
        "Stage counts:",
        stage_counts,
    )

    print(
        "Planned fit/adaptation operations:",
        planned_fit_operations,
    )

    print(
        "Execution-plan leakage checks: PASSED"
    )

    print(
        "Real model fitting performed: False"
    )

    print(
        "Performance observed: False"
    )

    print(
        "Created:",
        REGISTRY_PATH.relative_to(
            ROOT
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


if __name__ == "__main__":
    main()
