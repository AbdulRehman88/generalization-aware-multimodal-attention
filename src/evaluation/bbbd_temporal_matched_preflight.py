"""No-fitting structural preflight for matched BBBD temporal sensitivity.

The preflight validates every duration, modality path, participant split,
training-only cleanup contract, and effective candidate count without fitting
a SHAP selector, classifier, or threshold.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from numbers import Integral
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from src.evaluation.bbbd_execution_plan import (
    enumerate_candidate_specs,
)
from src.evaluation.bbbd_protocol import (
    load_bbbd_protocol_config,
)
from src.evaluation.bbbd_runner import (
    feature_columns,
    load_path_table,
    select_participants,
    training_only_feature_cleanup,
)


DURATIONS = (4, 8, 16, 24, 32)

DURATION_DIRECTORIES = {
    4: "window_04s_matched_387",
    8: "window_08s_matched_387",
    16: "window_16s_matched_387",
    24: "window_24s_matched_387",
    32: "window_32s_matched_387",
}

PLAN_FILES = {
    4: "bbbd_temporal_matched_execution_plan_04s_v1.json",
    8: "bbbd_temporal_matched_execution_plan_08s_v1.json",
    16: "bbbd_temporal_matched_execution_plan_16s_v1.json",
    24: "bbbd_temporal_matched_execution_plan_24s_v1.json",
    32: "bbbd_temporal_matched_execution_plan_32s_v1.json",
}

EXPECTED_FEATURE_COUNTS = {
    "EEG": 232,
    "ECG": 16,
    "Pupil": 23,
    "ECG_EEG": 248,
    "ECG_Pupil": 39,
    "EEG_Pupil": 255,
    "ECG_EEG_Pupil": 271,
}

EXPECTED_CANDIDATE_COUNTS = {
    "EEG": 8,
    "ECG": 2,
    "Pupil": 4,
    "ECG_EEG": 8,
    "ECG_Pupil": 4,
    "EEG_Pupil": 8,
    "ECG_EEG_Pupil": 8,
}


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)

    if path.is_absolute():
        return path

    return Path.cwd() / path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for block in iter(
            lambda: stream.read(4 * 1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def sha256_json(value: Any) -> str:
    return hashlib.sha256(
        canonical_json(value).encode("utf-8")
    ).hexdigest()


def unwrap_loaded_path_table(value: Any) -> pd.DataFrame:
    """Extract the DataFrame returned by the BBBD path loader.

    The locked runner currently returns a tuple containing the feature
    DataFrame and companion metadata. This helper also permits a direct
    DataFrame return so the preflight remains compatible with either
    interface without changing the production runner.
    """

    if isinstance(value, pd.DataFrame):
        return value

    if isinstance(value, tuple):
        frames = [
            item
            for item in value
            if isinstance(item, pd.DataFrame)
        ]

        if len(frames) == 1:
            return frames[0]

        raise TypeError(
            "load_path_table tuple must contain exactly one DataFrame; "
            f"observed {len(frames)}."
        )

    raise TypeError(
        "Unsupported load_path_table return type: "
        f"{type(value)!r}"
    )


def string_sequence(value: Any) -> list[str] | None:
    if isinstance(value, str):
        return None

    if not isinstance(value, Sequence):
        return None

    converted = list(value)

    if all(isinstance(item, str) for item in converted):
        return converted

    return None


def nested_mappings(value: Any) -> list[Mapping[str, Any]]:
    mappings: list[Mapping[str, Any]] = []

    def visit(item: Any) -> None:
        if isinstance(item, Mapping):
            mappings.append(item)

            for child in item.values():
                visit(child)

        elif isinstance(item, Sequence) and not isinstance(
            item,
            (str, bytes),
        ):
            for child in item:
                visit(child)

        elif hasattr(item, "__dict__"):
            visit(vars(item))

    visit(value)

    return mappings


def find_named_string_sequence(
    value: Any,
    token: str,
) -> list[str] | None:
    token = token.lower()

    for mapping in nested_mappings(value):
        for key, item in mapping.items():
            if token not in str(key).lower():
                continue

            sequence = string_sequence(item)

            if sequence is not None:
                return sequence

    return None


def find_named_integer(
    value: Any,
    token: str,
) -> int | None:
    """Find an integer cleanup count in nested audit metadata."""

    token = token.lower()
    preferred: list[int] = []
    fallback: list[int] = []

    for mapping in nested_mappings(value):
        for key, item in mapping.items():
            key_text = str(key).lower()

            if token not in key_text:
                continue

            if isinstance(item, bool):
                continue

            if not isinstance(item, Integral):
                continue

            observed = int(item)

            if observed < 0:
                raise RuntimeError(
                    f"Negative cleanup count for {key}: {observed}"
                )

            if "count" in key_text:
                preferred.append(observed)
            else:
                fallback.append(observed)

    values = preferred or fallback

    if not values:
        return None

    if len(set(values)) != 1:
        raise RuntimeError(
            f"Conflicting cleanup counts for {token}: {values}"
        )

    return values[0]


def direct_string_sequences(value: Any) -> list[list[str]]:
    sequences: list[list[str]] = []

    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes),
    ):
        for item in value:
            sequence = string_sequence(item)

            if sequence is not None:
                sequences.append(sequence)

    return sequences


def normalize_cleanup_result(
    result: Any,
    candidate_features: Sequence[str],
) -> dict[str, Any]:
    """Normalize production cleanup output without requiring name lists.

    The production cleanup contract may expose retained feature names plus
    numeric constant/duplicate counts rather than explicit removed-name
    lists. Exact removed-feature accounting remains mandatory.
    """

    candidate_list = list(candidate_features)
    candidate_set = set(candidate_list)

    retained = find_named_string_sequence(
        result,
        "retained",
    )

    constant_names = find_named_string_sequence(
        result,
        "constant",
    )

    duplicate_names = find_named_string_sequence(
        result,
        "duplicate",
    )

    sequences = direct_string_sequences(result)

    valid_sequences = [
        sequence
        for sequence in sequences
        if set(sequence) <= candidate_set
    ]

    if retained is None and valid_sequences:
        retained = max(
            valid_sequences,
            key=len,
        )

    if retained is None:
        raise RuntimeError(
            "Could not identify retained features from "
            "training_only_feature_cleanup output. "
            f"Return type: {type(result)!r}"
        )

    retained = list(retained)

    if not retained:
        raise RuntimeError(
            "Training cleanup retained no features."
        )

    if len(retained) != len(set(retained)):
        raise RuntimeError(
            "Training cleanup returned duplicate retained-feature names."
        )

    if not set(retained) <= candidate_set:
        raise RuntimeError(
            "Training cleanup returned an unknown feature."
        )

    removed_set = candidate_set - set(retained)

    constant_names = (
        []
        if constant_names is None
        else list(constant_names)
    )

    duplicate_names = (
        []
        if duplicate_names is None
        else list(duplicate_names)
    )

    named_removed = (
        set(constant_names)
        | set(duplicate_names)
    )

    if not named_removed <= removed_set:
        raise RuntimeError(
            "Cleanup audit identifies a removed feature as retained."
        )

    if set(constant_names) & set(duplicate_names):
        raise RuntimeError(
            "A feature is classified as both constant and duplicate."
        )

    constant_count = find_named_integer(
        result,
        "constant",
    )

    duplicate_count = find_named_integer(
        result,
        "duplicate",
    )

    if constant_count is None and constant_names:
        constant_count = len(constant_names)

    if duplicate_count is None and duplicate_names:
        duplicate_count = len(duplicate_names)

    if constant_count is None and duplicate_count is not None:
        constant_count = (
            len(removed_set)
            - duplicate_count
        )

    if duplicate_count is None and constant_count is not None:
        duplicate_count = (
            len(removed_set)
            - constant_count
        )

    if constant_count is None and duplicate_count is None:
        # Constant/duplicate subtype metadata is optional in the locked
        # production cleanup interface. Never invent the subtype.
        constant_count = len(constant_names)
        duplicate_count = len(duplicate_names)

    if constant_count < len(constant_names):
        raise RuntimeError(
            "Constant-feature count is smaller than its explicit name list."
        )

    if duplicate_count < len(duplicate_names):
        raise RuntimeError(
            "Duplicate-feature count is smaller than its explicit name list."
        )

    classified_removed_count = int(
        constant_count
        + duplicate_count
    )

    total_removed_count = int(
        len(removed_set)
    )

    if classified_removed_count > total_removed_count:
        raise RuntimeError(
            "Cleanup subtype counts exceed the total removed-feature count: "
            f"removed={total_removed_count}, "
            f"constant={constant_count}, "
            f"duplicate={duplicate_count}."
        )

    unclassified_removed_count = int(
        total_removed_count
        - classified_removed_count
    )

    unnamed_removed_features = sorted(
        removed_set - named_removed
    )

    return {
        "retained_features":
            retained,
        "constant_features":
            constant_names,
        "duplicate_features":
            duplicate_names,
        "unnamed_removed_features":
            unnamed_removed_features,
        "original_feature_count":
            len(candidate_list),
        "retained_feature_count":
            len(retained),
        "removed_feature_count":
            total_removed_count,
        "constant_feature_count":
            int(constant_count),
        "duplicate_feature_count":
            int(duplicate_count),
        "unclassified_removed_feature_count":
            unclassified_removed_count,
        "cleanup_subtype_breakdown_complete":
            unclassified_removed_count == 0,
    }


def candidate_specs_for_count(
    evaluation_config: Mapping[str, Any],
    feature_count: int,
) -> list[dict[str, Any]]:
    signature = inspect.signature(
        enumerate_candidate_specs
    )

    parameter_count = len(
        signature.parameters
    )

    if (
        "evaluation_config" in signature.parameters
        and "feature_count" in signature.parameters
    ):
        result = enumerate_candidate_specs(
            evaluation_config=evaluation_config,
            feature_count=int(feature_count),
        )

    else:
        raise RuntimeError(
            "Unexpected enumerate_candidate_specs signature: "
            f"{signature}"
        )

    if not isinstance(result, list):
        result = list(result)

    return result


def role_sets(frame: pd.DataFrame) -> dict[str, set[str]]:
    return {
        "participants":
            set(
                frame[
                    "participant"
                ].astype(str)
            ),
        "recordings":
            set(
                frame[
                    "recording_id"
                ].astype(str)
            ),
        "segments":
            set(
                frame[
                    "segment_id"
                ].astype(str)
            ),
    }


def validate_role_partition(
    full_frame: pd.DataFrame,
    roles: Mapping[str, pd.DataFrame],
) -> None:
    if not roles:
        raise RuntimeError(
            "Role mapping is empty."
        )

    role_names = list(roles)
    role_membership = {
        name:
            role_sets(frame)
        for name, frame in roles.items()
    }

    for name, frame in roles.items():
        if frame.empty:
            raise RuntimeError(
                f"Evaluation role is empty: {name}"
            )

        if frame[
            "segment_id"
        ].duplicated().any():
            raise RuntimeError(
                f"Role contains duplicate segments: {name}"
            )

    for left_index, left_name in enumerate(role_names):
        for right_name in role_names[
            left_index + 1:
        ]:
            for unit in [
                "participants",
                "recordings",
                "segments",
            ]:
                overlap = (
                    role_membership[
                        left_name
                    ][
                        unit
                    ]
                    & role_membership[
                        right_name
                    ][
                        unit
                    ]
                )

                if overlap:
                    raise RuntimeError(
                        f"Role overlap for {unit}: "
                        f"{left_name} versus {right_name}"
                    )

    combined_segments = set().union(
        *[
            membership[
                "segments"
            ]
            for membership
            in role_membership.values()
        ]
    )

    full_segments = set(
        full_frame[
            "segment_id"
        ].astype(str)
    )

    if combined_segments != full_segments:
        missing = full_segments - combined_segments
        extra = combined_segments - full_segments

        raise RuntimeError(
            "Role partition is incomplete. "
            f"Missing={len(missing)}, extra={len(extra)}"
        )


def training_cleanup_contract(
    training_frame: pd.DataFrame,
    candidate_features: Sequence[str],
    evaluation_config: Mapping[str, Any],
) -> dict[str, Any]:
    cleanup_result = training_only_feature_cleanup(
        training_frame,
        list(candidate_features),
    )

    cleanup = normalize_cleanup_result(
        cleanup_result,
        candidate_features,
    )

    candidate_specs = candidate_specs_for_count(
        evaluation_config,
        cleanup[
            "retained_feature_count"
        ],
    )

    candidate_ids = [
        str(specification["candidate_id"])
        for specification in candidate_specs
    ]

    if len(candidate_ids) != len(set(candidate_ids)):
        raise RuntimeError(
            "Cleanup-derived candidate identifiers are not unique."
        )

    cleanup[
        "candidate_count"
    ] = len(candidate_specs)

    cleanup[
        "candidate_id_digest"
    ] = sha256_json(
        candidate_ids
    )

    return cleanup


def run_preflight(
    *,
    config_path: str | Path,
    matched_root: str | Path,
    plan_root: str | Path,
    performance_root: str | Path,
) -> tuple[dict[str, Any], pd.DataFrame]:
    resolved_config = resolve_project_path(
        config_path
    )

    resolved_matched_root = resolve_project_path(
        matched_root
    )

    resolved_plan_root = resolve_project_path(
        plan_root
    )

    resolved_performance_root = resolve_project_path(
        performance_root
    )

    if resolved_performance_root.exists():
        raise RuntimeError(
            "Temporal performance output exists before preflight."
        )

    for path in [
        resolved_config,
        resolved_matched_root,
        resolved_plan_root,
    ]:
        if not path.exists():
            raise FileNotFoundError(path)

    config = load_bbbd_protocol_config(
        resolved_config
    )

    evaluation_config = config[
        "evaluation"
    ]

    contract_records: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    duration_summaries: list[dict[str, Any]] = []
    source_hashes: dict[str, str] = {}

    total_contracts = 0
    total_candidate_fits = 0
    candidate_mismatches = 0

    for duration in DURATIONS:
        duration_root = (
            resolved_matched_root
            / DURATION_DIRECTORIES[
                duration
            ]
        )

        plan_path = (
            resolved_plan_root
            / PLAN_FILES[
                duration
            ]
        )

        if not duration_root.is_dir():
            raise FileNotFoundError(
                duration_root
            )

        if not plan_path.is_file():
            raise FileNotFoundError(
                plan_path
            )

        plan = json.loads(
            plan_path.read_text(
                encoding="utf-8"
            )
        )

        context = plan[
            "temporal_sensitivity_context"
        ]

        if int(
            context[
                "duration_seconds"
            ]
        ) != duration:
            raise RuntimeError(
                f"{duration}s plan context mismatch."
            )

        if int(
            context[
                "recordings"
            ]
        ) != 387:
            raise RuntimeError(
                f"{duration}s plan does not use 387 recordings."
            )

        if int(
            context[
                "participants"
            ]
        ) != 36:
            raise RuntimeError(
                f"{duration}s plan does not use 36 participants."
            )

        if plan.get(
            "performance_observed"
        ) is not False:
            raise RuntimeError(
                f"{duration}s plan reports observed performance."
            )

        if plan.get(
            "feature_selector_fitted"
        ) is not False:
            raise RuntimeError(
                f"{duration}s plan reports a fitted selector."
            )

        if plan.get(
            "classifier_fitted"
        ) is not False:
            raise RuntimeError(
                f"{duration}s plan reports a fitted classifier."
            )

        if plan[
            "feature_counts"
        ] != EXPECTED_FEATURE_COUNTS:
            raise RuntimeError(
                f"{duration}s feature-count contract changed."
            )

        if plan[
            "candidate_counts_per_path"
        ] != EXPECTED_CANDIDATE_COUNTS:
            raise RuntimeError(
                f"{duration}s candidate-count contract changed."
            )

        source_hashes[
            plan_path.as_posix()
        ] = sha256_file(
            plan_path
        )

        duration_contracts = 0
        duration_candidate_fits = 0

        for path_name in plan[
            "paths"
        ]:
            expected_feature_count = int(
                plan[
                    "feature_counts"
                ][
                    path_name
                ]
            )

            table = unwrap_loaded_path_table(
                load_path_table(
                    duration_root,
                    path_name,
                    expected_feature_count,
                )
            )

            if table[
                "segment_id"
            ].duplicated().any():
                raise RuntimeError(
                    f"{duration}s/{path_name} contains duplicate segments."
                )

            if table[
                "recording_id"
            ].nunique() != 387:
                raise RuntimeError(
                    f"{duration}s/{path_name} does not contain "
                    "387 recordings."
                )

            if table[
                "participant"
            ].nunique() != 36:
                raise RuntimeError(
                    f"{duration}s/{path_name} does not contain "
                    "36 participants."
                )

            candidates = feature_columns(
                table
            )

            if len(candidates) != expected_feature_count:
                raise RuntimeError(
                    f"{duration}s/{path_name} feature count mismatch."
                )

            path_contracts: list[dict[str, Any]] = []

            for fold in plan[
                "within_experiment_folds"
            ]:
                dataset = str(
                    fold[
                        "dataset"
                    ]
                )

                dataset_frame = table.loc[
                    table[
                        "dataset"
                    ].astype(str)
                    == dataset
                ].copy()

                training = select_participants(
                    dataset_frame,
                    fold[
                        "inner_training_participants"
                    ],
                )

                validation = select_participants(
                    dataset_frame,
                    fold[
                        "inner_validation_participants"
                    ],
                )

                test = select_participants(
                    dataset_frame,
                    [
                        fold[
                            "outer_test_participant"
                        ]
                    ],
                )

                validate_role_partition(
                    dataset_frame,
                    {
                        "inner_training":
                            training,
                        "inner_validation":
                            validation,
                        "outer_test":
                            test,
                    },
                )

                cleanup = training_cleanup_contract(
                    training,
                    candidates,
                    evaluation_config,
                )

                expected_candidates = int(
                    plan[
                        "candidate_counts_per_path"
                    ][
                        path_name
                    ]
                )

                mismatch = (
                    cleanup[
                        "candidate_count"
                    ]
                    != expected_candidates
                )

                candidate_mismatches += int(
                    mismatch
                )

                if mismatch:
                    raise RuntimeError(
                        f"{duration}s/{path_name}/within fold "
                        f"{fold['outer_fold_index']} candidate-count "
                        "mismatch."
                    )

                record = {
                    "duration_seconds":
                        duration,
                    "path":
                        path_name,
                    "split_type":
                        "within_experiment",
                    "split_identity":
                        (
                            f"{dataset}::outer_"
                            f"{int(fold['outer_fold_index']):02d}"
                        ),
                    "training_participants":
                        int(
                            training[
                                "participant"
                            ].nunique()
                        ),
                    "validation_participants":
                        int(
                            validation[
                                "participant"
                            ].nunique()
                        ),
                    "test_participants":
                        int(
                            test[
                                "participant"
                            ].nunique()
                        ),
                    **cleanup,
                }

                path_contracts.append(
                    record
                )

                contract_records.append(
                    record
                )

            for direction in plan[
                "cross_experiment_directions"
            ]:
                source = str(
                    direction[
                        "source"
                    ]
                )

                target = str(
                    direction[
                        "target"
                    ]
                )

                source_frame = table.loc[
                    table[
                        "dataset"
                    ].astype(str)
                    == source
                ].copy()

                target_frame = table.loc[
                    table[
                        "dataset"
                    ].astype(str)
                    == target
                ].copy()

                training = select_participants(
                    source_frame,
                    direction[
                        "source_training_participants"
                    ],
                )

                validation = select_participants(
                    source_frame,
                    direction[
                        "source_validation_participants"
                    ],
                )

                validate_role_partition(
                    source_frame,
                    {
                        "source_training":
                            training,
                        "source_validation":
                            validation,
                    },
                )

                if target_frame.empty:
                    raise RuntimeError(
                        f"{duration}s/{path_name}/{target} target is empty."
                    )

                source_participants = set(
                    source_frame[
                        "participant"
                    ].astype(str)
                )

                target_participants = set(
                    target_frame[
                        "participant"
                    ].astype(str)
                )

                if source_participants & target_participants:
                    raise RuntimeError(
                        "Cross-experiment participant identities overlap."
                    )

                source_recordings = set(
                    source_frame[
                        "recording_id"
                    ].astype(str)
                )

                target_recordings = set(
                    target_frame[
                        "recording_id"
                    ].astype(str)
                )

                if source_recordings & target_recordings:
                    raise RuntimeError(
                        "Cross-experiment recording identities overlap."
                    )

                source_segments = set(
                    source_frame[
                        "segment_id"
                    ].astype(str)
                )

                target_segments = set(
                    target_frame[
                        "segment_id"
                    ].astype(str)
                )

                if source_segments & target_segments:
                    raise RuntimeError(
                        "Cross-experiment segment identities overlap."
                    )

                cleanup = training_cleanup_contract(
                    training,
                    candidates,
                    evaluation_config,
                )

                expected_candidates = int(
                    plan[
                        "candidate_counts_per_path"
                    ][
                        path_name
                    ]
                )

                mismatch = (
                    cleanup[
                        "candidate_count"
                    ]
                    != expected_candidates
                )

                candidate_mismatches += int(
                    mismatch
                )

                if mismatch:
                    raise RuntimeError(
                        f"{duration}s/{path_name}/{direction['name']} "
                        "candidate-count mismatch."
                    )

                record = {
                    "duration_seconds":
                        duration,
                    "path":
                        path_name,
                    "split_type":
                        "cross_experiment",
                    "split_identity":
                        str(
                            direction[
                                "name"
                            ]
                        ),
                    "training_participants":
                        int(
                            training[
                                "participant"
                            ].nunique()
                        ),
                    "validation_participants":
                        int(
                            validation[
                                "participant"
                            ].nunique()
                        ),
                    "test_participants":
                        int(
                            target_frame[
                                "participant"
                            ].nunique()
                        ),
                    **cleanup,
                }

                path_contracts.append(
                    record
                )

                contract_records.append(
                    record
                )

            if len(path_contracts) != 38:
                raise RuntimeError(
                    f"{duration}s/{path_name} has "
                    f"{len(path_contracts)} contracts instead of 38."
                )

            path_candidate_fits = int(
                sum(
                    record[
                        "candidate_count"
                    ]
                    for record in path_contracts
                )
            )

            expected_path_candidate_fits = (
                int(
                    plan[
                        "candidate_counts_per_path"
                    ][
                        path_name
                    ]
                )
                * 38
            )

            if (
                path_candidate_fits
                != expected_path_candidate_fits
            ):
                raise RuntimeError(
                    f"{duration}s/{path_name} candidate-fit total changed."
                )

            original_counts = [
                record[
                    "original_feature_count"
                ]
                for record in path_contracts
            ]

            retained_counts = [
                record[
                    "retained_feature_count"
                ]
                for record in path_contracts
            ]

            removed_counts = [
                record[
                    "removed_feature_count"
                ]
                for record in path_contracts
            ]

            constant_counts = [
                record[
                    "constant_feature_count"
                ]
                for record in path_contracts
            ]

            duplicate_counts = [
                record[
                    "duplicate_feature_count"
                ]
                for record in path_contracts
            ]

            unclassified_removed_counts = [
                record[
                    "unclassified_removed_feature_count"
                ]
                for record in path_contracts
            ]

            subtype_breakdown_flags = [
                bool(
                    record[
                        "cleanup_subtype_breakdown_complete"
                    ]
                )
                for record in path_contracts
            ]

            candidate_counts = [
                record[
                    "candidate_count"
                ]
                for record in path_contracts
            ]

            summary_rows.append(
                {
                    "duration_seconds":
                        duration,
                    "path":
                        path_name,
                    "contracts_checked":
                        len(path_contracts),
                    "original_feature_count_min":
                        min(original_counts),
                    "original_feature_count_max":
                        max(original_counts),
                    "retained_feature_count_min":
                        min(retained_counts),
                    "retained_feature_count_max":
                        max(retained_counts),
                    "removed_feature_count_min":
                        min(removed_counts),
                    "removed_feature_count_max":
                        max(removed_counts),
                    "constant_feature_count_min":
                        min(constant_counts),
                    "constant_feature_count_max":
                        max(constant_counts),
                    "duplicate_feature_count_min":
                        min(duplicate_counts),
                    "duplicate_feature_count_max":
                        max(duplicate_counts),
                    "unclassified_removed_feature_count_min":
                        min(unclassified_removed_counts),
                    "unclassified_removed_feature_count_max":
                        max(unclassified_removed_counts),
                    "cleanup_subtype_breakdown_complete_all":
                        all(subtype_breakdown_flags),
                    "candidate_count_min":
                        min(candidate_counts),
                    "candidate_count_max":
                        max(candidate_counts),
                    "candidate_fits":
                        path_candidate_fits,
                    "role_contracts_passed":
                        True,
                    "candidate_contracts_passed":
                        True,
                }
            )

            duration_contracts += len(
                path_contracts
            )

            duration_candidate_fits += (
                path_candidate_fits
            )

            print(
                f"{duration:>2}s | "
                f"{path_name:<14} | "
                f"contracts={len(path_contracts):>2} | "
                f"candidate_fits={path_candidate_fits:>3} | "
                f"retained={min(retained_counts)}"
                f"-{max(retained_counts)}"
            )

        if duration_contracts != 266:
            raise RuntimeError(
                f"{duration}s has {duration_contracts} contracts "
                "instead of 266."
            )

        if duration_candidate_fits != 1596:
            raise RuntimeError(
                f"{duration}s has {duration_candidate_fits} candidate fits "
                "instead of 1596."
            )

        duration_total_fits = (
            duration_candidate_fits
            + duration_contracts
            + duration_contracts
        )

        if duration_total_fits != 2128:
            raise RuntimeError(
                f"{duration}s total planned fits changed."
            )

        duration_summaries.append(
            {
                "duration_seconds":
                    duration,
                "path_split_contracts_checked":
                    duration_contracts,
                "candidate_model_fits":
                    duration_candidate_fits,
                "shap_selector_fits":
                    duration_contracts,
                "selected_candidate_final_refits":
                    duration_contracts,
                "total_estimator_fits":
                    duration_total_fits,
                "performance_observed":
                    False,
                "feature_selector_fitted":
                    False,
                "classifier_fitted":
                    False,
            }
        )

        total_contracts += duration_contracts
        total_candidate_fits += duration_candidate_fits

    if total_contracts != 1330:
        raise RuntimeError(
            f"Observed {total_contracts} total contracts instead of 1330."
        )

    if total_candidate_fits != 7980:
        raise RuntimeError(
            "Observed candidate-fit total differs from 7,980."
        )

    total_selector_fits = total_contracts
    total_final_refits = total_contracts

    total_estimator_fits = (
        total_candidate_fits
        + total_selector_fits
        + total_final_refits
    )

    if total_estimator_fits != 10640:
        raise RuntimeError(
            "Total planned estimator-fit count differs from 10,640."
        )

    if candidate_mismatches != 0:
        raise RuntimeError(
            "One or more candidate-count contracts failed."
        )

    payload = {
        "audit_identity":
            "bbbd_temporal_matched_preflight_v1",
        "source_commit":
            "a457947",
        "status":
            "passed",
        "scientific_role":
            "no-fitting structural preflight for matched temporal sensitivity",
        "durations_seconds":
            list(DURATIONS),
        "participants":
            36,
        "recordings":
            387,
        "paths":
            7,
        "evaluation_splits_per_duration":
            38,
        "path_split_contracts_checked":
            total_contracts,
        "within_experiment_path_folds":
            1260,
        "cross_experiment_path_directions":
            70,
        "candidate_model_fits":
            total_candidate_fits,
        "shap_selector_fits":
            total_selector_fits,
        "selected_candidate_final_refits":
            total_final_refits,
        "total_estimator_fits":
            total_estimator_fits,
        "candidate_count_mismatches":
            candidate_mismatches,
        "role_contracts_passed":
            True,
        "training_only_cleanup_checked":
            True,
        "performance_observed":
            False,
        "feature_selector_fitted":
            False,
        "classifier_fitted":
            False,
        "threshold_selected":
            False,
        "primary_four_second_392_recording_result_unchanged":
            True,
        "matched_four_second_role":
            "sensitivity reference only",
        "contract_digest_sha256":
            sha256_json(
                contract_records
            ),
        "duration_summaries":
            duration_summaries,
        "source_hashes":
            source_hashes,
        "cleanup_function_signature":
            str(
                inspect.signature(
                    training_only_feature_cleanup
                )
            ),
        "candidate_function_signature":
            str(
                inspect.signature(
                    enumerate_candidate_specs
                )
            ),
    }

    return payload, pd.DataFrame(
        summary_rows
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the no-fitting matched BBBD temporal preflight."
        )
    )

    parser.add_argument(
        "--config",
        default="configs/bbbd.yaml",
    )

    parser.add_argument(
        "--matched-root",
        default=(
            "outputs/revision/features/"
            "bbbd_temporal_matched_v1"
        ),
    )

    parser.add_argument(
        "--plan-root",
        default="_research_audit",
    )

    parser.add_argument(
        "--performance-root",
        default=(
            "outputs/revision/evaluation/"
            "bbbd_temporal_matched_v1"
        ),
    )

    parser.add_argument(
        "--output-json",
        required=True,
    )

    parser.add_argument(
        "--output-csv",
        required=True,
    )

    arguments = parser.parse_args()

    payload, summary = run_preflight(
        config_path=arguments.config,
        matched_root=arguments.matched_root,
        plan_root=arguments.plan_root,
        performance_root=arguments.performance_root,
    )

    output_json = resolve_project_path(
        arguments.output_json
    )

    output_csv = resolve_project_path(
        arguments.output_csv
    )

    output_json.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_json.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    summary.to_csv(
        output_csv,
        index=False,
    )

    print(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
