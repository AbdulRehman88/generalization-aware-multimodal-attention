"""Prespecified BBBD evaluation execution-plan construction.

This module enumerates and validates evaluation roles and fit counts. It does
not fit feature selectors or classifiers and does not inspect performance.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from src.evaluation.bbbd_protocol import (
    deterministic_inner_split,
    deterministic_source_split,
    load_bbbd_protocol_config,
    validate_role_disjointness,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

METADATA_COLUMNS = [
    "segment_id",
    "recording_id",
    "dataset",
    "participant",
    "subject",
    "session",
    "task",
    "label",
    "condition",
    "window_index",
    "start_seconds",
    "end_seconds",
    "pupil_valid_fraction",
]


def resolve_project_path(
    value: str | Path,
) -> Path:
    """Resolve a project-relative path."""

    path = Path(value)

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path.resolve()


def expand_parameter_grid(
    parameters: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Expand list-valued model parameters deterministically."""

    fixed = {}
    variable_names = []
    variable_values = []

    for name in sorted(
        parameters
    ):
        value = parameters[
            name
        ]

        if isinstance(
            value,
            list,
        ):
            if not value:
                raise ValueError(
                    f"Parameter grid is empty for {name}."
                )

            variable_names.append(
                str(
                    name
                )
            )

            variable_values.append(
                list(
                    value
                )
            )
        else:
            fixed[
                str(
                    name
                )
            ] = value

    combinations = []

    for values in itertools.product(
        *variable_values
    ):
        candidate = dict(
            fixed
        )

        for name, value in zip(
            variable_names,
            values,
        ):
            candidate[
                name
            ] = value

        combinations.append(
            candidate
        )

    if not variable_names:
        combinations = [
            dict(
                fixed
            )
        ]

    canonical = {}

    for candidate in combinations:
        key = json.dumps(
            candidate,
            sort_keys=True,
            default=str,
        )

        canonical[
            key
        ] = candidate

    return [
        canonical[
            key
        ]
        for key in sorted(
            canonical
        )
    ]


def effective_top_k_candidates(
    feature_count: int,
    configured_values: Sequence[int | str],
) -> list[dict[str, Any]]:
    """Collapse configured top-k values to unique effective feature counts."""

    feature_count = int(
        feature_count
    )

    if feature_count < 1:
        raise ValueError(
            "Feature count must be positive."
        )

    candidates = {}

    for configured in configured_values:
        if isinstance(
            configured,
            str,
        ):
            if configured.lower() != "all":
                raise ValueError(
                    f"Unknown top-k value: {configured}"
                )

            effective = feature_count
            configured_label = "all"

        else:
            configured_integer = int(
                configured
            )

            if configured_integer < 1:
                raise ValueError(
                    "Numerical top-k values must be positive."
                )

            effective = min(
                configured_integer,
                feature_count,
            )

            configured_label = configured_integer

        if effective not in candidates:
            candidates[
                effective
            ] = {
                "configured_top_k":
                    configured_label,
                "effective_top_k":
                    effective,
            }
        else:
            previous = candidates[
                effective
            ][
                "configured_top_k"
            ]

            if previous == "all":
                continue

            if configured_label == "all":
                candidates[
                    effective
                ][
                    "configured_top_k"
                ] = "all"

    return [
        candidates[
            effective
        ]
        for effective in sorted(
            candidates
        )
    ]


def enumerate_candidate_specs(
    evaluation_config: Mapping[str, Any],
    *,
    feature_count: int,
) -> list[dict[str, Any]]:
    """Enumerate unique model and effective-feature candidates."""

    models = evaluation_config[
        "training"
    ][
        "models"
    ]

    top_k_values = evaluation_config[
        "feature_selection"
    ][
        "top_k_values"
    ]

    effective_top_k = effective_top_k_candidates(
        feature_count,
        top_k_values,
    )

    candidates = []

    for model_name in sorted(
        models
    ):
        parameter_grid = expand_parameter_grid(
            models[
                model_name
            ]
        )

        for parameter_index, parameters in enumerate(
            parameter_grid
        ):
            for top_k in effective_top_k:
                candidate = {
                    "model":
                        str(
                            model_name
                        ),
                    "parameter_index":
                        int(
                            parameter_index
                        ),
                    "parameters":
                        parameters,
                    **top_k,
                }

                candidate[
                    "candidate_id"
                ] = hashlib.sha256(
                    json.dumps(
                        candidate,
                        sort_keys=True,
                        default=str,
                    ).encode(
                        "utf-8"
                    )
                ).hexdigest()[
                    :16
                ]

                candidates.append(
                    candidate
                )

    candidate_ids = [
        candidate[
            "candidate_id"
        ]
        for candidate in candidates
    ]

    if len(
        candidate_ids
    ) != len(
        set(
            candidate_ids
        )
    ):
        raise RuntimeError(
            "Candidate identifiers are not unique."
        )

    return candidates


def _role_summary(
    frame: pd.DataFrame,
) -> dict[str, Any]:
    """Summarize one participant-contained role."""

    recording_frame = frame[
        [
            "recording_id",
            "participant",
            "dataset",
            "task",
            "label",
        ]
    ].drop_duplicates(
        "recording_id"
    )

    label_counts = (
        recording_frame[
            "label"
        ]
        .astype(int)
        .value_counts()
        .sort_index()
        .to_dict()
    )

    if set(
        label_counts
    ) != {
        0,
        1,
    }:
        raise RuntimeError(
            "An evaluation role does not contain both labels."
        )

    return {
        "participants":
            int(
                frame[
                    "participant"
                ].nunique()
            ),
        "recordings":
            int(
                recording_frame[
                    "recording_id"
                ].nunique()
            ),
        "windows":
            int(
                len(
                    frame
                )
            ),
        "recording_label_counts": {
            str(
                int(
                    key
                )
            ):
                int(
                    value
                )
            for key, value
            in label_counts.items()
        },
    }


def _metadata_role(
    metadata: pd.DataFrame,
    participants: Sequence[str],
) -> pd.DataFrame:
    """Select metadata for an exact participant role."""

    participant_set = {
        str(
            participant
        )
        for participant in participants
    }

    return metadata.loc[
        metadata[
            "participant"
        ].astype(str).isin(
            participant_set
        )
    ].copy()


def build_execution_plan(
    metadata: pd.DataFrame,
    config: Mapping[str, Any],
    feature_counts: Mapping[str, int],
) -> dict[str, Any]:
    """Build the complete no-training BBBD execution plan."""

    required = set(
        METADATA_COLUMNS
    )

    missing = required - set(
        metadata.columns
    )

    if missing:
        raise ValueError(
            f"Metadata lacks columns: {sorted(missing)}"
        )

    if metadata[
        "segment_id"
    ].astype(str).duplicated().any():
        raise RuntimeError(
            "Duplicate segment IDs detected."
        )

    evaluation = config[
        "evaluation"
    ]

    paths = [
        str(
            path
        )
        for path in evaluation[
            "direct_paths"
        ]
    ]

    if set(
        paths
    ) != set(
        feature_counts
    ):
        raise RuntimeError(
            "Configured paths and feature-count paths differ."
        )

    candidate_specs = {
        path:
            enumerate_candidate_specs(
                evaluation,
                feature_count=
                    int(
                        feature_counts[
                            path
                        ]
                    ),
            )
        for path in paths
    }

    candidate_counts = {
        path:
            int(
                len(
                    candidates
                )
            )
        for path, candidates
        in candidate_specs.items()
    }

    effective_top_k = {
        path:
            [
                int(
                    candidate[
                        "effective_top_k"
                    ]
                )
                for candidate
                in candidate_specs[
                    path
                ]
                if candidate[
                    "model"
                ] == sorted(
                    evaluation[
                        "training"
                    ][
                        "models"
                    ]
                )[
                    0
                ]
                and candidate[
                    "parameter_index"
                ] == 0
            ]
        for path in paths
    }

    within_config = evaluation[
        "within_experiment"
    ]

    within_folds = []

    for dataset_name in within_config[
        "experiments"
    ]:
        dataset_frame = metadata.loc[
            metadata[
                "dataset"
            ].astype(str)
            == str(
                dataset_name
            )
        ].copy()

        participants = sorted(
            dataset_frame[
                "participant"
            ].astype(str).unique()
        )

        for outer_index, outer_participant in enumerate(
            participants
        ):
            training_participants, validation_participants = (
                deterministic_inner_split(
                    participants,
                    outer_test_participant=
                        outer_participant,
                    random_seed=
                        int(
                            within_config[
                                "split_seed"
                            ]
                        ),
                    validation_fraction=
                        float(
                            within_config[
                                "inner_validation_fraction"
                            ]
                        ),
                    minimum_validation_participants=
                        int(
                            within_config[
                                "minimum_inner_validation_participants"
                            ]
                        ),
                )
            )

            role_frames = {
                "inner_training":
                    _metadata_role(
                        dataset_frame,
                        training_participants,
                    ),
                "inner_validation":
                    _metadata_role(
                        dataset_frame,
                        validation_participants,
                    ),
                "outer_test":
                    _metadata_role(
                        dataset_frame,
                        [
                            outer_participant,
                        ],
                    ),
            }

            validate_role_disjointness(
                role_frames
            )

            within_folds.append(
                {
                    "dataset":
                        str(
                            dataset_name
                        ),
                    "outer_fold_index":
                        int(
                            outer_index
                        ),
                    "outer_test_participant":
                        str(
                            outer_participant
                        ),
                    "inner_training_participants":
                        list(
                            training_participants
                        ),
                    "inner_validation_participants":
                        list(
                            validation_participants
                        ),
                    "roles": {
                        role:
                            _role_summary(
                                frame
                            )
                        for role, frame
                        in role_frames.items()
                    },
                    "candidate_fits_across_paths":
                        int(
                            sum(
                                candidate_counts.values()
                            )
                        ),
                    "selector_fits_across_paths":
                        int(
                            len(
                                paths
                            )
                        ),
                    "final_refits_across_paths":
                        int(
                            len(
                                paths
                            )
                        ),
                }
            )

    cross_config = evaluation[
        "cross_experiment"
    ]

    cross_directions = []

    for direction in cross_config[
        "directions"
    ]:
        source_name = str(
            direction[
                "source"
            ]
        )

        target_name = str(
            direction[
                "target"
            ]
        )

        source_frame = metadata.loc[
            metadata[
                "dataset"
            ].astype(str)
            == source_name
        ].copy()

        target_frame = metadata.loc[
            metadata[
                "dataset"
            ].astype(str)
            == target_name
        ].copy()

        source_participants = sorted(
            source_frame[
                "participant"
            ].astype(str).unique()
        )

        source_training, source_validation = (
            deterministic_source_split(
                source_participants,
                random_seed=
                    int(
                        cross_config[
                            "split_seed"
                        ]
                    ),
                validation_fraction=
                    float(
                        cross_config[
                            "source_validation_fraction"
                        ]
                    ),
                minimum_validation_participants=
                    int(
                        cross_config[
                            "minimum_source_validation_participants"
                        ]
                    ),
            )
        )

        role_frames = {
            "source_training":
                _metadata_role(
                    source_frame,
                    source_training,
                ),
            "source_validation":
                _metadata_role(
                    source_frame,
                    source_validation,
                ),
            "target_test":
                target_frame.copy(),
        }

        validate_role_disjointness(
            role_frames
        )

        cross_directions.append(
            {
                "name":
                    str(
                        direction[
                            "name"
                        ]
                    ),
                "source":
                    source_name,
                "target":
                    target_name,
                "source_training_participants":
                    list(
                        source_training
                    ),
                "source_validation_participants":
                    list(
                        source_validation
                    ),
                "roles": {
                    role:
                        _role_summary(
                            frame
                        )
                    for role, frame
                    in role_frames.items()
                },
                "candidate_fits_across_paths":
                    int(
                        sum(
                            candidate_counts.values()
                        )
                    ),
                "selector_fits_across_paths":
                    int(
                        len(
                            paths
                        )
                    ),
                "final_refits_across_paths":
                    int(
                        len(
                            paths
                        )
                    ),
                "target_tuning_permitted":
                    False,
            }
        )

    within_candidate_fits = int(
        sum(
            fold[
                "candidate_fits_across_paths"
            ]
            for fold in within_folds
        )
    )

    within_selector_fits = int(
        sum(
            fold[
                "selector_fits_across_paths"
            ]
            for fold in within_folds
        )
    )

    within_final_refits = int(
        sum(
            fold[
                "final_refits_across_paths"
            ]
            for fold in within_folds
        )
    )

    cross_candidate_fits = int(
        sum(
            direction[
                "candidate_fits_across_paths"
            ]
            for direction in cross_directions
        )
    )

    cross_selector_fits = int(
        sum(
            direction[
                "selector_fits_across_paths"
            ]
            for direction in cross_directions
        )
    )

    cross_final_refits = int(
        sum(
            direction[
                "final_refits_across_paths"
            ]
            for direction in cross_directions
        )
    )

    totals = {
        "within_outer_folds":
            int(
                len(
                    within_folds
                )
            ),
        "cross_experiment_directions":
            int(
                len(
                    cross_directions
                )
            ),
        "candidate_model_fits":
            (
                within_candidate_fits
                + cross_candidate_fits
            ),
        "shap_selector_fits":
            (
                within_selector_fits
                + cross_selector_fits
            ),
        "selected_candidate_final_refits":
            (
                within_final_refits
                + cross_final_refits
            ),
    }

    totals[
        "total_estimator_fits"
    ] = int(
        totals[
            "candidate_model_fits"
        ]
        + totals[
            "shap_selector_fits"
        ]
        + totals[
            "selected_candidate_final_refits"
        ]
    )

    return {
        "protocol_name":
            str(
                evaluation[
                    "protocol_name"
                ]
            ),
        "plan_version":
            int(
                evaluation[
                    "execution"
                ][
                    "plan_version"
                ]
            ),
        "performance_observed":
            False,
        "classifier_fitted":
            False,
        "feature_selector_fitted":
            False,
        "paths":
            paths,
        "feature_counts": {
            path:
                int(
                    feature_counts[
                        path
                    ]
                )
            for path in paths
        },
        "effective_top_k":
            effective_top_k,
        "candidate_counts_per_path":
            candidate_counts,
        "candidate_specs":
            candidate_specs,
        "within_experiment_folds":
            within_folds,
        "cross_experiment_directions":
            cross_directions,
        "fit_totals":
            totals,
        "safeguards": {
            "participant_disjoint_roles":
                True,
            "recording_disjoint_roles":
                True,
            "segment_disjoint_roles":
                True,
            "source_only_cross_experiment_selection":
                True,
            "target_experiment_tuning":
                False,
            "deduplicated_effective_top_k":
                True,
            "outer_test_grid_revision":
                False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build the no-training BBBD evaluation execution plan."
        )
    )

    parser.add_argument(
        "--config",
        default="configs/bbbd.yaml",
    )

    parser.add_argument(
        "--feature-root",
        default=(
            "outputs/revision/features/"
            "bbbd_primary"
        ),
    )

    parser.add_argument(
        "--output-json",
        default=(
            "_research_audit/"
            "bbbd_evaluation_execution_plan_v1.json"
        ),
    )

    arguments = parser.parse_args()

    config = load_bbbd_protocol_config(
        arguments.config
    )

    feature_root = resolve_project_path(
        arguments.feature_root
    )

    summary_path = (
        feature_root
        / "summary.json"
    )

    metadata_path = (
        feature_root
        / "ECG_EEG_Pupil_features.csv"
    )

    if not summary_path.is_file():
        raise FileNotFoundError(
            summary_path
        )

    if not metadata_path.is_file():
        raise FileNotFoundError(
            metadata_path
        )

    summary = json.loads(
        summary_path.read_text(
            encoding="utf-8"
        )
    )

    feature_counts = {
        str(
            key
        ):
            int(
                value
            )
        for key, value
        in summary[
            "validation"
        ][
            "feature_counts"
        ].items()
    }

    metadata = pd.read_csv(
        metadata_path,
        usecols=
            METADATA_COLUMNS,
        low_memory=False,
    )

    plan = build_execution_plan(
        metadata,
        config,
        feature_counts,
    )

    output_path = resolve_project_path(
        arguments.output_json
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            plan,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "paths":
                    plan[
                        "paths"
                    ],
                "effective_top_k":
                    plan[
                        "effective_top_k"
                    ],
                "candidate_counts_per_path":
                    plan[
                        "candidate_counts_per_path"
                    ],
                "fit_totals":
                    plan[
                        "fit_totals"
                    ],
                "within_outer_folds":
                    len(
                        plan[
                            "within_experiment_folds"
                        ]
                    ),
                "cross_directions":
                    len(
                        plan[
                            "cross_experiment_directions"
                        ]
                    ),
                "classifier_fitted":
                    plan[
                        "classifier_fitted"
                    ],
                "performance_observed":
                    plan[
                        "performance_observed"
                    ],
            },
            indent=2,
            sort_keys=True,
        )
    )

    print(
        "Created:",
        output_path
    )


if __name__ == "__main__":
    main()