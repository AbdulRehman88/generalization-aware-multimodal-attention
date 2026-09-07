"""Leakage-safe BBBD evaluation-protocol utilities.

This module defines and validates the evaluation contract. It does not train
classifiers or inspect outer-test performance.
"""

from __future__ import annotations

import hashlib
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

REQUIRED_PREDICTION_COLUMNS = {
    "dataset",
    "participant",
    "recording_id",
    "task",
    "label",
}


def resolve_project_path(
    value: str | Path,
) -> Path:
    """Resolve a project-relative path."""

    path = Path(value)

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path.resolve()


def load_bbbd_protocol_config(
    path: str | Path = "configs/bbbd.yaml",
) -> dict[str, Any]:
    """Load and validate the locked BBBD evaluation configuration."""

    resolved = resolve_project_path(
        path
    )

    if not resolved.is_file():
        raise FileNotFoundError(
            resolved
        )

    config = yaml.safe_load(
        resolved.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(
        config,
        dict,
    ):
        raise TypeError(
            "BBBD configuration must be a mapping."
        )

    if "evaluation" not in config:
        raise ValueError(
            "BBBD evaluation section is missing."
        )

    evaluation = config[
        "evaluation"
    ]

    required_sections = {
        "within_experiment",
        "cross_experiment",
        "training",
        "feature_selection",
        "threshold_selection",
        "aggregation",
        "metrics",
        "bootstrap",
        "selection_objective",
        "stop_rule",
    }

    missing = (
        required_sections
        - set(
            evaluation
        )
    )

    if missing:
        raise ValueError(
            "BBBD evaluation configuration lacks sections: "
            f"{sorted(missing)}"
        )

    return config


def _unique_strings(
    values: Iterable[Any],
) -> list[str]:
    """Return sorted unique nonempty string values."""

    result = sorted(
        {
            str(value)
            for value in values
            if str(value)
        }
    )

    if not result:
        raise ValueError(
            "Expected at least one identifier."
        )

    return result


def deterministic_inner_split(
    participants: Sequence[str],
    *,
    outer_test_participant: str,
    random_seed: int,
    validation_fraction: float,
    minimum_validation_participants: int,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Create a deterministic participant-disjoint inner split."""

    all_participants = _unique_strings(
        participants
    )

    outer_test = str(
        outer_test_participant
    )

    if outer_test not in all_participants:
        raise ValueError(
            "Outer-test participant is absent from the participant set."
        )

    candidates = [
        participant
        for participant in all_participants
        if participant != outer_test
    ]

    if len(
        candidates
    ) < 3:
        raise ValueError(
            "At least three non-test participants are required."
        )

    validation_fraction = float(
        validation_fraction
    )

    if not (
        0.0
        < validation_fraction
        < 1.0
    ):
        raise ValueError(
            "Validation fraction must be in (0, 1)."
        )

    minimum_validation_participants = int(
        minimum_validation_participants
    )

    if minimum_validation_participants < 1:
        raise ValueError(
            "At least one validation participant is required."
        )

    validation_count = max(
        minimum_validation_participants,
        int(
            round(
                len(
                    candidates
                )
                * validation_fraction
            )
        ),
    )

    validation_count = min(
        validation_count,
        len(
            candidates
        )
        - 1,
    )

    ranked = sorted(
        candidates,
        key=lambda participant: (
            hashlib.sha256(
                (
                    f"{int(random_seed)}|"
                    f"{outer_test}|"
                    f"{participant}"
                ).encode(
                    "utf-8"
                )
            ).hexdigest(),
            participant,
        ),
    )

    validation = tuple(
        sorted(
            ranked[
                :validation_count
            ]
        )
    )

    training = tuple(
        sorted(
            ranked[
                validation_count:
            ]
        )
    )

    if set(
        training
    ) & set(
        validation
    ):
        raise RuntimeError(
            "Inner training and validation participants overlap."
        )

    if outer_test in training or outer_test in validation:
        raise RuntimeError(
            "Outer-test participant entered an inner role."
        )

    if (
        set(
            training
        )
        | set(
            validation
        )
        | {
            outer_test
        }
    ) != set(
        all_participants
    ):
        raise RuntimeError(
            "Participant split does not cover the full participant set."
        )

    return training, validation



def deterministic_source_split(
    participants: Sequence[str],
    *,
    random_seed: int,
    validation_fraction: float,
    minimum_validation_participants: int,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Create a deterministic source-only train/validation split."""

    all_participants = _unique_strings(
        participants
    )

    if len(
        all_participants
    ) < 3:
        raise ValueError(
            "At least three source participants are required."
        )

    validation_fraction = float(
        validation_fraction
    )

    if not (
        0.0
        < validation_fraction
        < 1.0
    ):
        raise ValueError(
            "Validation fraction must be in (0, 1)."
        )

    minimum_validation_participants = int(
        minimum_validation_participants
    )

    if minimum_validation_participants < 1:
        raise ValueError(
            "At least one validation participant is required."
        )

    validation_count = max(
        minimum_validation_participants,
        int(
            round(
                len(
                    all_participants
                )
                * validation_fraction
            )
        ),
    )

    validation_count = min(
        validation_count,
        len(
            all_participants
        )
        - 1,
    )

    ranked = sorted(
        all_participants,
        key=lambda participant: (
            hashlib.sha256(
                (
                    f"{int(random_seed)}|source-only|"
                    f"{participant}"
                ).encode(
                    "utf-8"
                )
            ).hexdigest(),
            participant,
        ),
    )

    validation = tuple(
        sorted(
            ranked[
                :validation_count
            ]
        )
    )

    training = tuple(
        sorted(
            ranked[
                validation_count:
            ]
        )
    )

    if set(
        training
    ) & set(
        validation
    ):
        raise RuntimeError(
            "Source training and validation participants overlap."
        )

    if (
        set(
            training
        )
        | set(
            validation
        )
    ) != set(
        all_participants
    ):
        raise RuntimeError(
            "Source split does not cover all source participants."
        )

    if not training or not validation:
        raise RuntimeError(
            "Source split produced an empty role."
        )

    return training, validation


def validate_role_disjointness(
    role_frames: Mapping[str, pd.DataFrame],
) -> None:
    """Verify participant, recording, and segment containment across roles."""

    if len(
        role_frames
    ) < 2:
        raise ValueError(
            "At least two roles are required."
        )

    role_sets: dict[
        str,
        dict[str, set[str]],
    ] = {}

    for role_name, frame in role_frames.items():
        required = {
            "participant",
            "recording_id",
            "segment_id",
        }

        missing = required - set(
            frame.columns
        )

        if missing:
            raise ValueError(
                f"{role_name}: missing role columns {sorted(missing)}"
            )

        if frame[
            "segment_id"
        ].astype(str).duplicated().any():
            raise ValueError(
                f"{role_name}: duplicate segment IDs detected."
            )

        role_sets[
            str(
                role_name
            )
        ] = {
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

    for first, second in combinations(
        sorted(
            role_sets
        ),
        2,
    ):
        for unit in [
            "participants",
            "recordings",
            "segments",
        ]:
            overlap = (
                role_sets[
                    first
                ][
                    unit
                ]
                & role_sets[
                    second
                ][
                    unit
                ]
            )

            if overlap:
                raise ValueError(
                    f"{first} and {second} overlap in {unit}: "
                    f"{sorted(overlap)[:10]}"
                )


def recording_equal_window_weights(
    frame: pd.DataFrame,
) -> np.ndarray:
    """Assign each recording total fitting weight one."""

    if "recording_id" not in frame.columns:
        raise ValueError(
            "recording_id is required for recording-equal weighting."
        )

    recording_ids = frame[
        "recording_id"
    ].astype(str)

    counts = recording_ids.map(
        recording_ids.value_counts()
    ).to_numpy(
        dtype=float
    )

    if np.any(
        counts <= 0
    ):
        raise RuntimeError(
            "Invalid recording window count."
        )

    weights = (
        1.0
        / counts
    )

    if not np.isfinite(
        weights
    ).all():
        raise RuntimeError(
            "Recording-equal weights are nonfinite."
        )

    check = pd.DataFrame(
        {
            "recording_id":
                recording_ids.to_numpy(),
            "weight":
                weights,
        }
    ).groupby(
        "recording_id"
    )[
        "weight"
    ].sum()

    if not np.allclose(
        check.to_numpy(
            dtype=float
        ),
        1.0,
        atol=1e-12,
        rtol=0.0,
    ):
        raise RuntimeError(
            "Each recording does not sum to total weight one."
        )

    return weights


def aggregate_recording_probabilities(
    frame: pd.DataFrame,
    *,
    probability_column: str = "probability",
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Aggregate window probabilities into recording-level decisions."""

    required = (
        REQUIRED_PREDICTION_COLUMNS
        | {
            probability_column,
        }
    )

    missing = required - set(
        frame.columns
    )

    if missing:
        raise ValueError(
            f"Prediction frame lacks columns: {sorted(missing)}"
        )

    probabilities = pd.to_numeric(
        frame[
            probability_column
        ],
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    if not np.isfinite(
        probabilities
    ).all():
        raise ValueError(
            "Window probabilities contain nonfinite values."
        )

    if np.any(
        (
            probabilities < 0.0
        )
        | (
            probabilities > 1.0
        )
    ):
        raise ValueError(
            "Probabilities must be in [0, 1]."
        )

    threshold = float(
        threshold
    )

    if not (
        0.0
        <= threshold
        <= 1.0
    ):
        raise ValueError(
            "Decision threshold must be in [0, 1]."
        )

    for column in [
        "dataset",
        "participant",
        "task",
        "label",
    ]:
        uniqueness = frame.groupby(
            "recording_id"
        )[
            column
        ].nunique(
            dropna=False
        )

        if not (
            uniqueness
            == 1
        ).all():
            raise ValueError(
                f"Recording metadata is inconsistent for {column}."
            )

    aggregated = (
        frame.groupby(
            "recording_id",
            as_index=False,
            sort=True,
        )
        .agg(
            dataset=(
                "dataset",
                "first",
            ),
            participant=(
                "participant",
                "first",
            ),
            task=(
                "task",
                "first",
            ),
            label=(
                "label",
                "first",
            ),
            probability=(
                probability_column,
                "mean",
            ),
            window_count=(
                probability_column,
                "size",
            ),
        )
    )

    aggregated[
        "label"
    ] = pd.to_numeric(
        aggregated[
            "label"
        ],
        errors="raise",
    ).astype(int)

    if not set(
        aggregated[
            "label"
        ].unique()
    ).issubset(
        {
            0,
            1,
        }
    ):
        raise ValueError(
            "Recording labels must be binary."
        )

    aggregated[
        "predicted_label"
    ] = (
        aggregated[
            "probability"
        ].astype(float)
        >= threshold
    ).astype(int)

    aggregated[
        "threshold"
    ] = threshold

    return aggregated


def binary_probability_metrics(
    labels: Sequence[int],
    probabilities: Sequence[float],
    *,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute prespecified binary probability metrics."""

    y_true = np.asarray(
        labels,
        dtype=int,
    ).reshape(-1)

    y_probability = np.asarray(
        probabilities,
        dtype=float,
    ).reshape(-1)

    if len(
        y_true
    ) != len(
        y_probability
    ):
        raise ValueError(
            "Label and probability lengths differ."
        )

    if len(
        y_true
    ) == 0:
        raise ValueError(
            "Metric input is empty."
        )

    if not set(
        np.unique(
            y_true
        )
    ).issubset(
        {
            0,
            1,
        }
    ):
        raise ValueError(
            "Labels must be binary."
        )

    if not np.isfinite(
        y_probability
    ).all():
        raise ValueError(
            "Probabilities contain nonfinite values."
        )

    if np.any(
        (
            y_probability < 0.0
        )
        | (
            y_probability > 1.0
        )
    ):
        raise ValueError(
            "Probabilities must be in [0, 1]."
        )

    threshold = float(
        threshold
    )

    y_predicted = (
        y_probability
        >= threshold
    ).astype(int)

    unique_labels = np.unique(
        y_true
    )

    if len(
        unique_labels
    ) == 2:
        roc_auc = float(
            roc_auc_score(
                y_true,
                y_probability,
            )
        )

        pr_auc = float(
            average_precision_score(
                y_true,
                y_probability,
            )
        )
    else:
        roc_auc = float(
            "nan"
        )

        pr_auc = float(
            "nan"
        )

    return {
        "accuracy":
            float(
                accuracy_score(
                    y_true,
                    y_predicted,
                )
            ),
        "balanced_accuracy":
            float(
                balanced_accuracy_score(
                    y_true,
                    y_predicted,
                )
            ),
        "macro_f1":
            float(
                f1_score(
                    y_true,
                    y_predicted,
                    average="macro",
                    labels=[
                        0,
                        1,
                    ],
                    zero_division=0,
                )
            ),
        "roc_auc":
            roc_auc,
        "pr_auc":
            pr_auc,
    }


def participant_metric_table(
    recording_predictions: pd.DataFrame,
    *,
    threshold: float,
) -> pd.DataFrame:
    """Compute participant-specific recording-level metrics."""

    required = {
        "participant",
        "label",
        "probability",
    }

    missing = required - set(
        recording_predictions.columns
    )

    if missing:
        raise ValueError(
            f"Recording predictions lack columns: {sorted(missing)}"
        )

    rows = []

    for participant, frame in recording_predictions.groupby(
        "participant",
        sort=True,
    ):
        metrics = binary_probability_metrics(
            frame[
                "label"
            ].astype(int),
            frame[
                "probability"
            ].astype(float),
            threshold=
                threshold,
        )

        rows.append(
            {
                "participant":
                    str(
                        participant
                    ),
                "recordings":
                    int(
                        len(
                            frame
                        )
                    ),
                **metrics,
            }
        )

    if not rows:
        raise ValueError(
            "No participant metrics were produced."
        )

    return pd.DataFrame(
        rows
    )


def participant_macro_metrics(
    recording_predictions: pd.DataFrame,
    *,
    threshold: float,
) -> dict[str, float]:
    """Macro-average participant-specific recording metrics."""

    table = participant_metric_table(
        recording_predictions,
        threshold=
            threshold,
    )

    result = {
        "participant_count":
            float(
                len(
                    table
                )
            ),
    }

    for metric in [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "roc_auc",
        "pr_auc",
    ]:
        values = pd.to_numeric(
            table[
                metric
            ],
            errors="coerce",
        ).to_numpy(
            dtype=float
        )

        finite = values[
            np.isfinite(
                values
            )
        ]

        result[
            metric
        ] = (
            float(
                np.mean(
                    finite
                )
            )
            if len(
                finite
            )
            else float(
                "nan"
            )
        )

    return result


def select_recording_threshold(
    validation_recording_predictions: pd.DataFrame,
    *,
    candidates: Sequence[float],
    primary_metric: str = "balanced_accuracy",
) -> tuple[float, pd.DataFrame]:
    """Select a threshold using validation recordings only."""

    required = {
        "label",
        "probability",
    }

    missing = required - set(
        validation_recording_predictions.columns
    )

    if missing:
        raise ValueError(
            f"Validation predictions lack columns: {sorted(missing)}"
        )

    candidate_values = sorted(
        {
            float(
                candidate
            )
            for candidate in candidates
        }
    )

    if not candidate_values:
        raise ValueError(
            "At least one threshold candidate is required."
        )

    if any(
        candidate < 0.0
        or candidate > 1.0
        for candidate in candidate_values
    ):
        raise ValueError(
            "Threshold candidates must be in [0, 1]."
        )

    rows = []

    for threshold in candidate_values:
        metrics = binary_probability_metrics(
            validation_recording_predictions[
                "label"
            ].astype(int),
            validation_recording_predictions[
                "probability"
            ].astype(float),
            threshold=
                threshold,
        )

        rows.append(
            {
                "threshold":
                    threshold,
                **metrics,
            }
        )

    table = pd.DataFrame(
        rows
    )

    if primary_metric not in table.columns:
        raise ValueError(
            f"Unknown threshold-selection metric: {primary_metric}"
        )

    ranked = table.assign(
        _distance_from_half=(
            table[
                "threshold"
            ]
            - 0.5
        ).abs()
    ).sort_values(
        [
            primary_metric,
            "_distance_from_half",
            "threshold",
        ],
        ascending=[
            False,
            True,
            True,
        ],
        kind="mergesort",
    )

    selected = float(
        ranked.iloc[
            0
        ][
            "threshold"
        ]
    )

    return (
        selected,
        table.sort_values(
            "threshold"
        ).reset_index(
            drop=True
        ),
    )


def participant_cluster_bootstrap(
    recording_predictions: pd.DataFrame,
    *,
    repetitions: int,
    random_seed: int,
    threshold: float,
) -> pd.DataFrame:
    """Bootstrap recording-level metrics by sampling participants."""

    required = {
        "participant",
        "label",
        "probability",
    }

    missing = required - set(
        recording_predictions.columns
    )

    if missing:
        raise ValueError(
            f"Recording predictions lack columns: {sorted(missing)}"
        )

    participants = _unique_strings(
        recording_predictions[
            "participant"
        ]
    )

    repetitions = int(
        repetitions
    )

    if repetitions < 1:
        raise ValueError(
            "Bootstrap repetitions must be positive."
        )

    participant_frames = {
        participant:
            recording_predictions.loc[
                recording_predictions[
                    "participant"
                ].astype(str)
                == participant
            ].copy()
        for participant in participants
    }

    random_generator = np.random.default_rng(
        int(
            random_seed
        )
    )

    rows = []

    for repetition in range(
        repetitions
    ):
        sampled = random_generator.choice(
            participants,
            size=len(
                participants
            ),
            replace=True,
        )

        sampled_frames = []

        for draw_index, participant in enumerate(
            sampled
        ):
            frame = participant_frames[
                str(
                    participant
                )
            ].copy()

            frame[
                "_bootstrap_participant"
            ] = (
                f"draw-{draw_index:05d}::"
                f"{participant}"
            )

            sampled_frames.append(
                frame
            )

        bootstrap_frame = pd.concat(
            sampled_frames,
            ignore_index=True,
        )

        pooled = binary_probability_metrics(
            bootstrap_frame[
                "label"
            ].astype(int),
            bootstrap_frame[
                "probability"
            ].astype(float),
            threshold=
                threshold,
        )

        participant_frame = bootstrap_frame.rename(
            columns={
                "_bootstrap_participant":
                    "original_participant",
                "participant":
                    "_source_participant",
            }
        ).rename(
            columns={
                "original_participant":
                    "participant",
            }
        )

        participant_macro = participant_macro_metrics(
            participant_frame,
            threshold=
                threshold,
        )

        row = {
            "repetition":
                repetition,
        }

        for metric, value in pooled.items():
            row[
                f"pooled_{metric}"
            ] = value

        for metric, value in participant_macro.items():
            row[
                f"participant_macro_{metric}"
            ] = value

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


def bootstrap_confidence_intervals(
    bootstrap: pd.DataFrame,
    *,
    confidence_level: float,
) -> pd.DataFrame:
    """Summarize percentile confidence intervals."""

    confidence_level = float(
        confidence_level
    )

    if not (
        0.0
        < confidence_level
        < 1.0
    ):
        raise ValueError(
            "Confidence level must be in (0, 1)."
        )

    alpha = (
        1.0
        - confidence_level
    )

    lower_percentile = (
        100.0
        * alpha
        / 2.0
    )

    upper_percentile = (
        100.0
        * (
            1.0
            - alpha / 2.0
        )
    )

    rows = []

    for column in bootstrap.columns:
        if column == "repetition":
            continue

        values = pd.to_numeric(
            bootstrap[
                column
            ],
            errors="coerce",
        ).to_numpy(
            dtype=float
        )

        finite = values[
            np.isfinite(
                values
            )
        ]

        if len(
            finite
        ) == 0:
            continue

        rows.append(
            {
                "metric":
                    column,
                "bootstrap_mean":
                    float(
                        np.mean(
                            finite
                        )
                    ),
                "ci_lower":
                    float(
                        np.percentile(
                            finite,
                            lower_percentile,
                        )
                    ),
                "ci_upper":
                    float(
                        np.percentile(
                            finite,
                            upper_percentile,
                        )
                    ),
                "finite_repetitions":
                    int(
                        len(
                            finite
                        )
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )