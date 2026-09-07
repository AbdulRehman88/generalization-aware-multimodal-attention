"""Leakage-safe, resumable BBBD evaluation runner.

The runner supports separate within-experiment nested LOSO and bidirectional
source-only cross-experiment transfer. All feature cleanup, SHAP ranking,
candidate selection, and threshold selection are restricted to the designated
training and validation roles.

Running this module with ``--synthetic-smoke`` fits synthetic data only.
Running it with ``--run-bbbd`` performs the locked BBBD evaluation.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import ExtraTreesClassifier
from xgboost import XGBClassifier

from src.evaluation.bbbd_execution_plan import (
    enumerate_candidate_specs,
)
from src.evaluation.bbbd_protocol import (
    aggregate_recording_probabilities,
    binary_probability_metrics,
    bootstrap_confidence_intervals,
    load_bbbd_protocol_config,
    participant_cluster_bootstrap,
    participant_macro_metrics,
    participant_metric_table,
    recording_equal_window_weights,
    select_recording_threshold,
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

PREDICTION_METADATA_COLUMNS = [
    "segment_id",
    "recording_id",
    "dataset",
    "participant",
    "task",
    "label",
]

MODEL_PRIORITY = {
    "extra_trees": 0,
    "xgboost": 1,
}


def resolve_project_path(
    value: str | Path,
) -> Path:
    """Resolve a path relative to the repository root."""

    path = Path(value)

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path.resolve()


def canonical_json(
    value: Any,
) -> str:
    """Serialize a value deterministically."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(
            ",",
            ":",
        ),
        default=str,
    )


def sha256_text(
    value: str,
) -> str:
    """Return a SHA-256 digest for text."""

    return hashlib.sha256(
        value.encode(
            "utf-8"
        )
    ).hexdigest()


def write_json_atomic(
    path: Path,
    value: Any,
) -> None:
    """Write JSON atomically."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_name(
        path.name
        + ".tmp"
    )

    temporary.write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )

    os.replace(
        temporary,
        path,
    )


def write_csv_atomic(
    path: Path,
    frame: pd.DataFrame,
) -> None:
    """Write a CSV file atomically."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_name(
        path.name
        + ".tmp"
    )

    frame.to_csv(
        temporary,
        index=False,
    )

    os.replace(
        temporary,
        path,
    )


def feature_columns(
    frame: pd.DataFrame,
) -> list[str]:
    """Return feature columns in their stored order."""

    features = [
        column
        for column in frame.columns
        if column not in METADATA_COLUMNS
    ]

    if not features:
        raise ValueError(
            "No feature columns were found."
        )

    if len(
        features
    ) != len(
        set(
            features
        )
    ):
        raise ValueError(
            "Duplicate feature names were detected."
        )

    return features


def validate_feature_frame(
    frame: pd.DataFrame,
    *,
    expected_feature_count: int | None = None,
) -> list[str]:
    """Validate one complete modality-path table."""

    required = set(
        METADATA_COLUMNS
    )

    missing = required - set(
        frame.columns
    )

    if missing:
        raise ValueError(
            f"Feature table lacks metadata columns: {sorted(missing)}"
        )

    if frame.empty:
        raise ValueError(
            "Feature table is empty."
        )

    if frame[
        "segment_id"
    ].astype(str).duplicated().any():
        raise ValueError(
            "Duplicate segment identifiers were detected."
        )

    features = feature_columns(
        frame
    )

    if (
        expected_feature_count is not None
        and len(
            features
        ) != int(
            expected_feature_count
        )
    ):
        raise ValueError(
            "Feature count differs from the locked contract: "
            f"{len(features)} versus {expected_feature_count}."
        )

    values = frame[
        features
    ].to_numpy(
        dtype=float
    )

    if not np.isfinite(
        values
    ).all():
        raise ValueError(
            "Feature table contains nonfinite numerical values."
        )

    labels = pd.to_numeric(
        frame[
            "label"
        ],
        errors="raise",
    ).astype(int)

    if not set(
        labels.unique()
    ).issubset(
        {
            0,
            1,
        }
    ):
        raise ValueError(
            "Feature-table labels are not binary."
        )

    return features


def training_only_feature_cleanup(
    training_frame: pd.DataFrame,
    candidate_features: Sequence[str],
) -> dict[str, Any]:
    """Remove constant and exactly duplicated features using training rows only."""

    features = [
        str(
            feature
        )
        for feature in candidate_features
    ]

    if not features:
        raise ValueError(
            "No candidate features were supplied."
        )

    values = training_frame[
        features
    ].to_numpy(
        dtype=np.float64,
    )

    if not np.isfinite(
        values
    ).all():
        raise ValueError(
            "Training features contain nonfinite values."
        )

    constant_features = []

    retained_after_constants = []

    for index, feature in enumerate(
        features
    ):
        column = values[
            :,
            index,
        ]

        if np.all(
            column
            == column[
                0
            ]
        ):
            constant_features.append(
                feature
            )
        else:
            retained_after_constants.append(
                feature
            )

    duplicate_features = []

    retained = []

    digest_to_feature: dict[
        str,
        str,
    ] = {}

    feature_arrays: dict[
        str,
        np.ndarray,
    ] = {}

    for feature in retained_after_constants:
        column = np.ascontiguousarray(
            training_frame[
                feature
            ].to_numpy(
                dtype=np.float64,
            )
        )

        digest = hashlib.sha256(
            column.tobytes()
        ).hexdigest()

        duplicate_of = None

        if digest in digest_to_feature:
            previous = digest_to_feature[
                digest
            ]

            if np.array_equal(
                column,
                feature_arrays[
                    previous
                ],
            ):
                duplicate_of = previous

        if duplicate_of is None:
            retained.append(
                feature
            )

            digest_to_feature[
                digest
            ] = feature

            feature_arrays[
                feature
            ] = column
        else:
            duplicate_features.append(
                {
                    "feature":
                        feature,
                    "duplicate_of":
                        duplicate_of,
                }
            )

    if not retained:
        raise RuntimeError(
            "Training-only cleanup removed every feature."
        )

    return {
        "input_feature_count":
            len(
                features
            ),
        "constant_features":
            constant_features,
        "duplicate_features":
            duplicate_features,
        "retained_features":
            retained,
        "retained_feature_count":
            len(
                retained
            ),
    }


def recording_balanced_sample_indices(
    frame: pd.DataFrame,
    *,
    maximum_rows: int,
    random_seed: int,
) -> np.ndarray:
    """Select deterministic approximately recording-balanced SHAP rows."""

    required = {
        "recording_id",
        "segment_id",
    }

    missing = required - set(
        frame.columns
    )

    if missing:
        raise ValueError(
            f"SHAP sample frame lacks columns: {sorted(missing)}"
        )

    maximum_rows = int(
        maximum_rows
    )

    if maximum_rows < 1:
        raise ValueError(
            "maximum_rows must be positive."
        )

    if len(
        frame
    ) <= maximum_rows:
        return np.arange(
            len(
                frame
            ),
            dtype=int,
        )

    groups: dict[
        str,
        list[int],
    ] = {}

    for recording_id, group in frame.groupby(
        "recording_id",
        sort=True,
    ):
        ranked = sorted(
            group.index.tolist(),
            key=lambda index: (
                sha256_text(
                    f"{int(random_seed)}|"
                    f"{frame.at[index, 'segment_id']}"
                ),
                int(
                    index
                ),
            ),
        )

        groups[
            str(
                recording_id
            )
        ] = [
            int(
                index
            )
            for index in ranked
        ]

    recording_ids = sorted(
        groups
    )

    selected = []

    depth = 0

    while len(
        selected
    ) < maximum_rows:
        added = False

        for recording_id in recording_ids:
            values = groups[
                recording_id
            ]

            if depth < len(
                values
            ):
                selected.append(
                    values[
                        depth
                    ]
                )

                added = True

                if len(
                    selected
                ) >= maximum_rows:
                    break

        if not added:
            break

        depth += 1

    if len(
        selected
    ) != maximum_rows:
        raise RuntimeError(
            "Could not construct the requested deterministic SHAP sample."
        )

    return np.asarray(
        selected,
        dtype=int,
    )


def normalize_binary_shap_values(
    values: Any,
    *,
    expected_rows: int,
    expected_features: int,
) -> np.ndarray:
    """Normalize binary TreeExplainer output across supported API shapes."""

    if isinstance(
        values,
        list,
    ):
        if len(
            values
        ) == 2:
            array = np.asarray(
                values[
                    1
                ],
                dtype=float,
            )
        elif len(
            values
        ) == 1:
            array = np.asarray(
                values[
                    0
                ],
                dtype=float,
            )
        else:
            raise RuntimeError(
                "Unexpected SHAP list length."
            )
    else:
        array = np.asarray(
            values,
            dtype=float,
        )

    if array.ndim == 3:
        if array.shape[
            2
        ] == 2:
            array = array[
                :,
                :,
                1,
            ]
        elif array.shape[
            2
        ] == 1:
            array = array[
                :,
                :,
                0,
            ]
        else:
            raise RuntimeError(
                f"Unexpected SHAP output shape: {array.shape}"
            )

    expected_shape = (
        int(
            expected_rows
        ),
        int(
            expected_features
        ),
    )

    if array.shape != expected_shape:
        raise RuntimeError(
            "Normalized SHAP shape differs from the expected contract: "
            f"{array.shape} versus {expected_shape}."
        )

    if not np.isfinite(
        array
    ).all():
        raise RuntimeError(
            "SHAP values contain nonfinite entries."
        )

    return array


def instantiate_estimator(
    model_name: str,
    parameters: Mapping[str, Any],
):
    """Instantiate one locked model candidate."""

    parameters = dict(
        parameters
    )

    if model_name == "extra_trees":
        return ExtraTreesClassifier(
            **parameters
        )

    if model_name == "xgboost":
        return XGBClassifier(
            **parameters
        )

    raise ValueError(
        f"Unknown model family: {model_name}"
    )


def positive_class_probability(
    model: Any,
    values: np.ndarray,
) -> np.ndarray:
    """Return the probability for binary class one."""

    probabilities = np.asarray(
        model.predict_proba(
            values
        ),
        dtype=float,
    )

    classes = np.asarray(
        model.classes_
    )

    matches = np.flatnonzero(
        classes
        == 1
    )

    if len(
        matches
    ) != 1:
        raise RuntimeError(
            f"Could not locate positive class in {classes.tolist()}."
        )

    result = probabilities[
        :,
        int(
            matches[
                0
            ]
        ),
    ]

    if result.ndim != 1:
        raise RuntimeError(
            "Positive-class probabilities must be one-dimensional."
        )

    if not np.isfinite(
        result
    ).all():
        raise RuntimeError(
            "Predicted probabilities contain nonfinite values."
        )

    if np.any(
        (
            result < 0.0
        )
        | (
            result > 1.0
        )
    ):
        raise RuntimeError(
            "Predicted probabilities fall outside [0, 1]."
        )

    return result


def fit_training_shap_ranking(
    training_frame: pd.DataFrame,
    retained_features: Sequence[str],
    *,
    selector_parameters: Mapping[str, Any],
    maximum_shap_samples: int,
    random_seed: int,
) -> tuple[pd.DataFrame, Any]:
    """Fit the selector and rank features using training participants only."""

    features = [
        str(
            feature
        )
        for feature in retained_features
    ]

    values = training_frame[
        features
    ].to_numpy(
        dtype=float
    )

    labels = training_frame[
        "label"
    ].to_numpy(
        dtype=int
    )

    weights = recording_equal_window_weights(
        training_frame
    )

    selector = XGBClassifier(
        **dict(
            selector_parameters
        )
    )

    selector.fit(
        values,
        labels,
        sample_weight=
            weights,
    )

    sample_positions = recording_balanced_sample_indices(
        training_frame.reset_index(
            drop=True
        ),
        maximum_rows=
            min(
                int(
                    maximum_shap_samples
                ),
                len(
                    training_frame
                ),
            ),
        random_seed=
            int(
                random_seed
            ),
    )

    sample_values = values[
        sample_positions
    ]

    explainer = shap.TreeExplainer(
        selector
    )

    raw_values = explainer.shap_values(
        sample_values
    )

    shap_values = normalize_binary_shap_values(
        raw_values,
        expected_rows=
            len(
                sample_values
            ),
        expected_features=
            len(
                features
            ),
    )

    importance = np.mean(
        np.abs(
            shap_values
        ),
        axis=0,
    )

    ranking = pd.DataFrame(
        {
            "feature":
                features,
            "mean_absolute_shap":
                importance,
        }
    ).sort_values(
        [
            "mean_absolute_shap",
            "feature",
        ],
        ascending=[
            False,
            True,
        ],
        kind="mergesort",
    ).reset_index(
        drop=True
    )

    ranking.insert(
        0,
        "rank",
        np.arange(
            1,
            len(
                ranking
            )
            + 1,
            dtype=int,
        ),
    )

    if ranking[
        "feature"
    ].duplicated().any():
        raise RuntimeError(
            "SHAP ranking contains duplicate features."
        )

    return ranking, selector


def window_prediction_frame(
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    *,
    role: str,
) -> pd.DataFrame:
    """Create a standardized window-prediction table."""

    if len(
        frame
    ) != len(
        probabilities
    ):
        raise ValueError(
            "Frame and probability lengths differ."
        )

    result = frame[
        PREDICTION_METADATA_COLUMNS
    ].copy()

    result[
        "probability"
    ] = np.asarray(
        probabilities,
        dtype=float,
    )

    result[
        "role"
    ] = str(
        role
    )

    return result


def candidate_rank_key(
    candidate_result: Mapping[str, Any],
) -> tuple[Any, ...]:
    """Apply the locked candidate-selection tie-breaking order."""

    return (
        -float(
            candidate_result[
                "validation_balanced_accuracy"
            ]
        ),
        -float(
            candidate_result[
                "validation_macro_f1"
            ]
        ),
        int(
            candidate_result[
                "effective_top_k"
            ]
        ),
        int(
            MODEL_PRIORITY[
                str(
                    candidate_result[
                        "model"
                    ]
                )
            ]
        ),
        canonical_json(
            candidate_result[
                "parameters"
            ]
        ),
        str(
            candidate_result[
                "candidate_id"
            ]
        ),
    )


def evaluate_candidate(
    training_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    ranking: pd.DataFrame,
    candidate: Mapping[str, Any],
    *,
    threshold_candidates: Sequence[float],
    threshold_metric: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Fit and score one validation candidate."""

    effective_top_k = int(
        candidate[
            "effective_top_k"
        ]
    )

    selected_features = ranking[
        "feature"
    ].astype(str).tolist()[
        :effective_top_k
    ]

    model = instantiate_estimator(
        str(
            candidate[
                "model"
            ]
        ),
        candidate[
            "parameters"
        ],
    )

    training_values = training_frame[
        selected_features
    ].to_numpy(
        dtype=float
    )

    validation_values = validation_frame[
        selected_features
    ].to_numpy(
        dtype=float
    )

    training_labels = training_frame[
        "label"
    ].to_numpy(
        dtype=int
    )

    training_weights = recording_equal_window_weights(
        training_frame
    )

    model.fit(
        training_values,
        training_labels,
        sample_weight=
            training_weights,
    )

    validation_probability = positive_class_probability(
        model,
        validation_values,
    )

    validation_windows = window_prediction_frame(
        validation_frame,
        validation_probability,
        role=
            "validation",
    )

    validation_recordings = aggregate_recording_probabilities(
        validation_windows,
        probability_column=
            "probability",
        threshold=
            0.5,
    )

    threshold, threshold_table = select_recording_threshold(
        validation_recordings,
        candidates=
            threshold_candidates,
        primary_metric=
            threshold_metric,
    )

    metrics = binary_probability_metrics(
        validation_recordings[
            "label"
        ],
        validation_recordings[
            "probability"
        ],
        threshold=
            threshold,
    )

    result = {
        "candidate_id":
            str(
                candidate[
                    "candidate_id"
                ]
            ),
        "model":
            str(
                candidate[
                    "model"
                ]
            ),
        "parameters":
            dict(
                candidate[
                    "parameters"
                ]
            ),
        "configured_top_k":
            candidate[
                "configured_top_k"
            ],
        "effective_top_k":
            effective_top_k,
        "selected_threshold":
            float(
                threshold
            ),
        "validation_recordings":
            int(
                len(
                    validation_recordings
                )
            ),
        "validation_balanced_accuracy":
            float(
                metrics[
                    "balanced_accuracy"
                ]
            ),
        "validation_macro_f1":
            float(
                metrics[
                    "macro_f1"
                ]
            ),
        "validation_accuracy":
            float(
                metrics[
                    "accuracy"
                ]
            ),
        "validation_roc_auc":
            float(
                metrics[
                    "roc_auc"
                ]
            ),
        "validation_pr_auc":
            float(
                metrics[
                    "pr_auc"
                ]
            ),
    }

    threshold_table = threshold_table.copy()

    threshold_table.insert(
        0,
        "candidate_id",
        result[
            "candidate_id"
        ],
    )

    return result, threshold_table


def evaluate_locked_split(
    *,
    training_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    candidate_features: Sequence[str],
    evaluation_config: Mapping[str, Any],
    output_directory: Path,
    split_identity: Mapping[str, Any],
    save_selected_model: bool,
) -> dict[str, Any]:
    """Evaluate one fully specified training/validation/test split."""

    validate_role_disjointness(
        {
            "training":
                training_frame,
            "validation":
                validation_frame,
            "test":
                test_frame,
        }
    )

    output_directory = Path(
        output_directory
    )

    completion_path = (
        output_directory
        / "complete.json"
    )

    split_hash = sha256_text(
        canonical_json(
            split_identity
        )
    )

    if completion_path.is_file():
        completion = json.loads(
            completion_path.read_text(
                encoding="utf-8"
            )
        )

        if completion.get(
            "split_sha256"
        ) != split_hash:
            raise RuntimeError(
                "Existing completed split does not match the requested split."
            )

        return completion

    if output_directory.exists():
        existing = list(
            output_directory.iterdir()
        )

        if existing:
            raise RuntimeError(
                "Incomplete nonempty split output exists: "
                f"{output_directory}"
            )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    started = time.time()

    cleanup = training_only_feature_cleanup(
        training_frame,
        candidate_features,
    )

    feature_config = evaluation_config[
        "feature_selection"
    ]

    selector_parameters = copy.deepcopy(
        evaluation_config[
            "training"
        ][
            "models"
        ][
            "xgboost"
        ]
    )

    for name, value in list(
        selector_parameters.items()
    ):
        if isinstance(
            value,
            list,
        ):
            if len(
                value
            ) != 1:
                raise RuntimeError(
                    f"Selector parameter {name} is not uniquely locked."
                )

            selector_parameters[
                name
            ] = value[
                0
            ]

    ranking, selector = fit_training_shap_ranking(
        training_frame,
        cleanup[
            "retained_features"
        ],
        selector_parameters=
            selector_parameters,
        maximum_shap_samples=
            int(
                feature_config[
                    "maximum_shap_samples"
                ]
            ),
        random_seed=
            int(
                feature_config[
                    "random_seed"
                ]
            ),
    )

    candidates = enumerate_candidate_specs(
        evaluation_config,
        feature_count=
            int(
                cleanup[
                    "retained_feature_count"
                ]
            ),
    )

    candidate_rows = []

    threshold_tables = []

    for candidate_index, candidate in enumerate(
        candidates,
        start=1,
    ):
        result, threshold_table = evaluate_candidate(
            training_frame,
            validation_frame,
            ranking,
            candidate,
            threshold_candidates=
                evaluation_config[
                    "threshold_selection"
                ][
                    "candidate_thresholds"
                ],
            threshold_metric=
                str(
                    evaluation_config[
                        "threshold_selection"
                    ][
                        "primary_metric"
                    ]
                ),
        )

        result[
            "candidate_index"
        ] = candidate_index

        candidate_rows.append(
            result
        )

        threshold_tables.append(
            threshold_table
        )

    selected = sorted(
        candidate_rows,
        key=
            candidate_rank_key,
    )[
        0
    ]

    selected_features = ranking[
        "feature"
    ].astype(str).tolist()[
        :int(
            selected[
                "effective_top_k"
            ]
        )
    ]

    final_training = pd.concat(
        [
            training_frame,
            validation_frame,
        ],
        ignore_index=True,
    )

    final_model = instantiate_estimator(
        str(
            selected[
                "model"
            ]
        ),
        selected[
            "parameters"
        ],
    )

    final_model.fit(
        final_training[
            selected_features
        ].to_numpy(
            dtype=float
        ),
        final_training[
            "label"
        ].to_numpy(
            dtype=int
        ),
        sample_weight=
            recording_equal_window_weights(
                final_training
            ),
    )

    test_probability = positive_class_probability(
        final_model,
        test_frame[
            selected_features
        ].to_numpy(
            dtype=float
        ),
    )

    test_windows = window_prediction_frame(
        test_frame,
        test_probability,
        role=
            "test",
    )

    threshold = float(
        selected[
            "selected_threshold"
        ]
    )

    test_recordings = aggregate_recording_probabilities(
        test_windows,
        probability_column=
            "probability",
        threshold=
            threshold,
    )

    recording_metrics = binary_probability_metrics(
        test_recordings[
            "label"
        ],
        test_recordings[
            "probability"
        ],
        threshold=
            threshold,
    )

    participant_metrics = participant_metric_table(
        test_recordings,
        threshold=
            threshold,
    )

    participant_macro = participant_macro_metrics(
        test_recordings,
        threshold=
            threshold,
    )

    window_metrics = binary_probability_metrics(
        test_windows[
            "label"
        ],
        test_windows[
            "probability"
        ],
        threshold=
            threshold,
    )

    candidate_table = pd.DataFrame(
        candidate_rows
    ).sort_values(
        [
            "validation_balanced_accuracy",
            "validation_macro_f1",
            "effective_top_k",
            "model",
            "candidate_id",
        ],
        ascending=[
            False,
            False,
            True,
            True,
            True,
        ],
        kind="mergesort",
    ).reset_index(
        drop=True
    )

    threshold_table = pd.concat(
        threshold_tables,
        ignore_index=True,
    )

    selected_payload = {
        **selected,
        "selected_features":
            selected_features,
        "selected_feature_count":
            len(
                selected_features
            ),
        "split_identity":
            dict(
                split_identity
            ),
        "split_sha256":
            split_hash,
    }

    metrics_payload = {
        "threshold":
            threshold,
        "recording_metrics":
            recording_metrics,
        "participant_macro_metrics":
            participant_macro,
        "window_metrics_descriptive":
            window_metrics,
        "training_participants":
            int(
                training_frame[
                    "participant"
                ].nunique()
            ),
        "validation_participants":
            int(
                validation_frame[
                    "participant"
                ].nunique()
            ),
        "test_participants":
            int(
                test_frame[
                    "participant"
                ].nunique()
            ),
        "training_recordings":
            int(
                training_frame[
                    "recording_id"
                ].nunique()
            ),
        "validation_recordings":
            int(
                validation_frame[
                    "recording_id"
                ].nunique()
            ),
        "test_recordings":
            int(
                test_recordings[
                    "recording_id"
                ].nunique()
            ),
        "training_windows":
            int(
                len(
                    training_frame
                )
            ),
        "validation_windows":
            int(
                len(
                    validation_frame
                )
            ),
        "test_windows":
            int(
                len(
                    test_frame
                )
            ),
    }

    write_csv_atomic(
        output_directory
        / "feature_ranking.csv",
        ranking,
    )

    write_json_atomic(
        output_directory
        / "feature_cleanup.json",
        cleanup,
    )

    write_csv_atomic(
        output_directory
        / "validation_candidates.csv",
        candidate_table,
    )

    write_csv_atomic(
        output_directory
        / "threshold_search.csv",
        threshold_table,
    )

    write_json_atomic(
        output_directory
        / "selected_candidate.json",
        selected_payload,
    )

    write_csv_atomic(
        output_directory
        / "window_predictions.csv",
        test_windows,
    )

    write_csv_atomic(
        output_directory
        / "recording_predictions.csv",
        test_recordings,
    )

    write_csv_atomic(
        output_directory
        / "participant_metrics.csv",
        participant_metrics,
    )

    write_json_atomic(
        output_directory
        / "metrics.json",
        metrics_payload,
    )

    if save_selected_model:
        joblib.dump(
            final_model,
            output_directory
            / "selected_model.joblib",
        )

        joblib.dump(
            selector,
            output_directory
            / "selector_model.joblib",
        )

    completion = {
        "completed":
            True,
        "split_identity":
            dict(
                split_identity
            ),
        "split_sha256":
            split_hash,
        "selected_model":
            str(
                selected[
                    "model"
                ]
            ),
        "selected_effective_top_k":
            int(
                selected[
                    "effective_top_k"
                ]
            ),
        "selected_threshold":
            threshold,
        "recording_metrics":
            recording_metrics,
        "participant_macro_metrics":
            participant_macro,
        "window_metrics_descriptive":
            window_metrics,
        "elapsed_seconds":
            float(
                time.time()
                - started
            ),
        "outer_or_target_performance_observed":
            True,
    }

    write_json_atomic(
        completion_path,
        completion,
    )

    return completion


def select_participants(
    frame: pd.DataFrame,
    participants: Sequence[str],
) -> pd.DataFrame:
    """Select exact participant-contained rows."""

    participant_set = {
        str(
            participant
        )
        for participant in participants
    }

    result = frame.loc[
        frame[
            "participant"
        ].astype(str).isin(
            participant_set
        )
    ].copy()

    observed = set(
        result[
            "participant"
        ].astype(str)
    )

    if observed != participant_set:
        raise RuntimeError(
            "Selected participant rows differ from the requested role."
        )

    return result


def load_path_table(
    feature_root: Path,
    path_name: str,
    expected_feature_count: int,
) -> tuple[pd.DataFrame, list[str]]:
    """Load and validate one locked BBBD path table."""

    path = (
        feature_root
        / f"{path_name}_features.csv"
    )

    if not path.is_file():
        raise FileNotFoundError(
            path
        )

    frame = pd.read_csv(
        path,
        low_memory=False,
    )

    features = validate_feature_frame(
        frame,
        expected_feature_count=
            expected_feature_count,
    )

    return frame, features


def protocol_summary_from_predictions(
    *,
    prediction_directories: Sequence[Path],
    output_directory: Path,
    bootstrap_repetitions: int,
    bootstrap_seed: int,
    confidence_level: float,
) -> dict[str, Any]:
    """Combine completed split predictions and compute protocol-level results."""

    recording_frames = []

    window_frames = []

    thresholds = []

    for directory in prediction_directories:
        completion_path = (
            directory
            / "complete.json"
        )

        recording_path = (
            directory
            / "recording_predictions.csv"
        )

        window_path = (
            directory
            / "window_predictions.csv"
        )

        if not completion_path.is_file():
            raise FileNotFoundError(
                completion_path
            )

        if not recording_path.is_file():
            raise FileNotFoundError(
                recording_path
            )

        if not window_path.is_file():
            raise FileNotFoundError(
                window_path
            )

        completion = json.loads(
            completion_path.read_text(
                encoding="utf-8"
            )
        )

        thresholds.append(
            float(
                completion[
                    "selected_threshold"
                ]
            )
        )

        recording_frames.append(
            pd.read_csv(
                recording_path,
                low_memory=False,
            )
        )

        window_frames.append(
            pd.read_csv(
                window_path,
                low_memory=False,
            )
        )

    recordings = pd.concat(
        recording_frames,
        ignore_index=True,
    )

    windows = pd.concat(
        window_frames,
        ignore_index=True,
    )

    if recordings[
        "recording_id"
    ].astype(str).duplicated().any():
        raise RuntimeError(
            "Combined protocol predictions contain duplicate recordings."
        )

    if windows[
        "segment_id"
    ].astype(str).duplicated().any():
        raise RuntimeError(
            "Combined protocol predictions contain duplicate windows."
        )

    recordings[
        "predicted_label"
    ] = (
        recordings[
            "probability"
        ].astype(float)
        >= recordings[
            "threshold"
        ].astype(float)
    ).astype(int)

    pooled_metrics = {
        "accuracy":
            float(
                np.mean(
                    recordings[
                        "predicted_label"
                    ].to_numpy(
                        dtype=int
                    )
                    == recordings[
                        "label"
                    ].to_numpy(
                        dtype=int
                    )
                )
            ),
        "balanced_accuracy":
            float(
                binary_probability_metrics(
                    recordings[
                        "label"
                    ],
                    recordings[
                        "probability"
                    ],
                    threshold=
                        0.5,
                )[
                    "balanced_accuracy"
                ]
            ),
    }

    from sklearn.metrics import (
        average_precision_score,
        balanced_accuracy_score,
        f1_score,
        roc_auc_score,
    )

    pooled_metrics = {
        "accuracy":
            float(
                np.mean(
                    recordings[
                        "predicted_label"
                    ].to_numpy(
                        dtype=int
                    )
                    == recordings[
                        "label"
                    ].to_numpy(
                        dtype=int
                    )
                )
            ),
        "balanced_accuracy":
            float(
                balanced_accuracy_score(
                    recordings[
                        "label"
                    ].astype(int),
                    recordings[
                        "predicted_label"
                    ].astype(int),
                )
            ),
        "macro_f1":
            float(
                f1_score(
                    recordings[
                        "label"
                    ].astype(int),
                    recordings[
                        "predicted_label"
                    ].astype(int),
                    average="macro",
                    labels=[
                        0,
                        1,
                    ],
                    zero_division=0,
                )
            ),
        "roc_auc":
            float(
                roc_auc_score(
                    recordings[
                        "label"
                    ].astype(int),
                    recordings[
                        "probability"
                    ].astype(float),
                )
            ),
        "pr_auc":
            float(
                average_precision_score(
                    recordings[
                        "label"
                    ].astype(int),
                    recordings[
                        "probability"
                    ].astype(float),
                )
            ),
    }

    participant_metrics = participant_metric_table(
        recordings,
        threshold=
            0.5,
    )

    participant_macro = {
        metric:
            float(
                participant_metrics[
                    metric
                ].dropna().mean()
            )
        for metric in [
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "roc_auc",
            "pr_auc",
        ]
    }

    bootstrap_input = recordings.copy()

    bootstrap_input[
        "probability"
    ] = (
        bootstrap_input[
            "probability"
        ].astype(float)
        - bootstrap_input[
            "threshold"
        ].astype(float)
        + 0.5
    ).clip(
        0.0,
        1.0,
    )

    bootstrap = participant_cluster_bootstrap(
        bootstrap_input,
        repetitions=
            int(
                bootstrap_repetitions
            ),
        random_seed=
            int(
                bootstrap_seed
            ),
        threshold=
            0.5,
    )

    confidence = bootstrap_confidence_intervals(
        bootstrap,
        confidence_level=
            float(
                confidence_level
            ),
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_csv_atomic(
        output_directory
        / "recording_predictions.csv",
        recordings,
    )

    write_csv_atomic(
        output_directory
        / "window_predictions.csv",
        windows,
    )

    write_csv_atomic(
        output_directory
        / "participant_metrics.csv",
        participant_metrics,
    )

    write_csv_atomic(
        output_directory
        / "participant_bootstrap.csv",
        bootstrap,
    )

    write_csv_atomic(
        output_directory
        / "confidence_intervals.csv",
        confidence,
    )

    summary = {
        "recordings":
            int(
                len(
                    recordings
                )
            ),
        "windows":
            int(
                len(
                    windows
                )
            ),
        "participants":
            int(
                recordings[
                    "participant"
                ].nunique()
            ),
        "fold_thresholds":
            thresholds,
        "pooled_recording_metrics":
            pooled_metrics,
        "participant_macro_metrics":
            participant_macro,
        "bootstrap_repetitions":
            int(
                bootstrap_repetitions
            ),
        "bootstrap_unit":
            "participant",
    }

    write_json_atomic(
        output_directory
        / "summary.json",
        summary,
    )

    return summary


def synthetic_table(
    *,
    datasets: Sequence[str],
    participants_per_dataset: int,
    random_seed: int,
    feature_count: int,
) -> pd.DataFrame:
    """Construct deterministic grouped synthetic data for integration testing."""

    random_generator = np.random.default_rng(
        int(
            random_seed
        )
    )

    feature_names = [
        f"feature_{index:03d}"
        for index in range(
            int(
                feature_count
            )
        )
    ]

    rows = []

    for dataset_index, dataset in enumerate(
        datasets
    ):
        dataset_shift = (
            0.15
            * dataset_index
        )

        for participant_index in range(
            int(
                participants_per_dataset
            )
        ):
            participant = (
                f"{dataset}::participant-"
                f"{participant_index:02d}"
            )

            participant_shift = random_generator.normal(
                0.0,
                0.25,
                size=
                    feature_count,
            )

            for label in [
                0,
                1,
            ]:
                recording_id = (
                    f"{participant}::recording-{label}"
                )

                for window_index in range(
                    6
                    + participant_index
                    % 3
                ):
                    values = random_generator.normal(
                        0.0,
                        0.7,
                        size=
                            feature_count,
                    )

                    values += participant_shift

                    values += dataset_shift

                    values[
                        0
                    ] += (
                        1.2
                        if label == 1
                        else -1.2
                    )

                    values[
                        1
                    ] += (
                        0.6
                        if label == 1
                        else -0.6
                    )

                    row = {
                        "segment_id":
                            (
                                f"{recording_id}::"
                                f"window-{window_index:03d}"
                            ),
                        "recording_id":
                            recording_id,
                        "dataset":
                            str(
                                dataset
                            ),
                        "participant":
                            participant,
                        "subject":
                            f"subject-{participant_index:02d}",
                        "session":
                            (
                                "ses-01"
                                if label == 1
                                else "ses-02"
                            ),
                        "task":
                            "stim01",
                        "label":
                            label,
                        "condition":
                            (
                                "attentive"
                                if label == 1
                                else "distracted"
                            ),
                        "window_index":
                            window_index,
                        "start_seconds":
                            float(
                                window_index
                                * 2
                            ),
                        "end_seconds":
                            float(
                                window_index
                                * 2
                                + 4
                            ),
                        "pupil_valid_fraction":
                            1.0,
                    }

                    row.update(
                        {
                            feature:
                                float(
                                    value
                                )
                            for feature, value
                            in zip(
                                feature_names,
                                values,
                            )
                        }
                    )

                    rows.append(
                        row
                    )

    return pd.DataFrame(
        rows
    )


def run_synthetic_smoke() -> dict[str, Any]:
    """Run one synthetic nested split and one synthetic transfer split."""

    config = load_bbbd_protocol_config()

    evaluation = copy.deepcopy(
        config[
            "evaluation"
        ]
    )

    evaluation[
        "training"
    ][
        "models"
    ][
        "extra_trees"
    ][
        "n_estimators"
    ] = [
        30,
    ]

    evaluation[
        "training"
    ][
        "models"
    ][
        "xgboost"
    ][
        "n_estimators"
    ] = [
        30,
    ]

    evaluation[
        "feature_selection"
    ][
        "top_k_values"
    ] = [
        3,
        "all",
    ]

    evaluation[
        "feature_selection"
    ][
        "maximum_shap_samples"
    ] = 100

    table = synthetic_table(
        datasets=[
            "synthetic_experiment2",
            "synthetic_experiment3",
        ],
        participants_per_dataset=
            6,
        random_seed=
            3407,
        feature_count=
            8,
    )

    features = feature_columns(
        table
    )

    experiment2 = table.loc[
        table[
            "dataset"
        ]
        == "synthetic_experiment2"
    ].copy()

    experiment3 = table.loc[
        table[
            "dataset"
        ]
        == "synthetic_experiment3"
    ].copy()

    participants = sorted(
        experiment2[
            "participant"
        ].unique()
    )

    nested_training = select_participants(
        experiment2,
        participants[
            :3
        ],
    )

    nested_validation = select_participants(
        experiment2,
        participants[
            3:5
        ],
    )

    nested_test = select_participants(
        experiment2,
        participants[
            5:
        ],
    )

    transfer_training = select_participants(
        experiment2,
        participants[
            :4
        ],
    )

    transfer_validation = select_participants(
        experiment2,
        participants[
            4:
        ],
    )

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(
            temporary
        )

        nested_completion = evaluate_locked_split(
            training_frame=
                nested_training,
            validation_frame=
                nested_validation,
            test_frame=
                nested_test,
            candidate_features=
                features,
            evaluation_config=
                evaluation,
            output_directory=
                root
                / "nested",
            split_identity={
                "protocol":
                    "synthetic_nested",
                "outer_test_participant":
                    participants[
                        5
                    ],
            },
            save_selected_model=
                True,
        )

        repeated_completion = evaluate_locked_split(
            training_frame=
                nested_training,
            validation_frame=
                nested_validation,
            test_frame=
                nested_test,
            candidate_features=
                features,
            evaluation_config=
                evaluation,
            output_directory=
                root
                / "nested",
            split_identity={
                "protocol":
                    "synthetic_nested",
                "outer_test_participant":
                    participants[
                        5
                    ],
            },
            save_selected_model=
                True,
        )

        if nested_completion != repeated_completion:
            raise RuntimeError(
                "Resume completion differs from the original completion."
            )

        transfer_completion = evaluate_locked_split(
            training_frame=
                transfer_training,
            validation_frame=
                transfer_validation,
            test_frame=
                experiment3,
            candidate_features=
                features,
            evaluation_config=
                evaluation,
            output_directory=
                root
                / "transfer",
            split_identity={
                "protocol":
                    "synthetic_transfer",
                "source":
                    "synthetic_experiment2",
                "target":
                    "synthetic_experiment3",
            },
            save_selected_model=
                True,
        )

        expected_files = {
            "complete.json",
            "feature_cleanup.json",
            "feature_ranking.csv",
            "metrics.json",
            "participant_metrics.csv",
            "recording_predictions.csv",
            "selected_candidate.json",
            "selected_model.joblib",
            "selector_model.joblib",
            "threshold_search.csv",
            "validation_candidates.csv",
            "window_predictions.csv",
        }

        for directory in [
            root
            / "nested",
            root
            / "transfer",
        ]:
            observed = {
                path.name
                for path in directory.iterdir()
                if path.is_file()
            }

            if observed != expected_files:
                raise RuntimeError(
                    "Synthetic runner output contract differs:\n"
                    f"{sorted(observed)}"
                )

    return {
        "audit_identity":
            "bbbd_runner_synthetic_smoke",
        "data_used":
            "synthetic_only",
        "bbbd_classifier_fitted":
            False,
        "outer_test_performance_observed":
            False,
        "target_experiment_performance_observed":
            False,
        "nested_split_completed":
            bool(
                nested_completion[
                    "completed"
                ]
            ),
        "transfer_split_completed":
            bool(
                transfer_completion[
                    "completed"
                ]
            ),
        "resume_verified":
            True,
        "recording_equal_weighting_verified":
            True,
        "training_only_cleanup_verified":
            True,
        "training_only_shap_verified":
            True,
        "validation_only_candidate_selection_verified":
            True,
        "validation_only_threshold_selection_verified":
            True,
        "output_contract_verified":
            True,
    }


def run_bbbd(
    *,
    config_path: str | Path,
    feature_root: str | Path,
    plan_path: str | Path,
    output_root: str | Path,
    path_limit: int | None,
    fold_limit: int | None,
) -> dict[str, Any]:
    """Run the locked BBBD evaluation."""

    config = load_bbbd_protocol_config(
        config_path
    )

    evaluation = config[
        "evaluation"
    ]

    feature_root = resolve_project_path(
        feature_root
    )

    plan_path = resolve_project_path(
        plan_path
    )

    output_root = resolve_project_path(
        output_root
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

    if bool(
        plan[
            "performance_observed"
        ]
    ):
        raise RuntimeError(
            "The locked pre-performance plan has been altered."
        )

    paths = list(
        plan[
            "paths"
        ]
    )

    if path_limit is not None:
        paths = paths[
            :int(
                path_limit
            )
        ]

    execution_rows = []

    for path_name in paths:
        frame, features = load_path_table(
            feature_root,
            path_name,
            expected_feature_count=
                int(
                    plan[
                        "feature_counts"
                    ][
                        path_name
                    ]
                ),
        )

        within_folds = list(
            plan[
                "within_experiment_folds"
            ]
        )

        if fold_limit is not None:
            within_folds = within_folds[
                :int(
                    fold_limit
                )
            ]

        for fold in within_folds:
            dataset_frame = frame.loc[
                frame[
                    "dataset"
                ].astype(str)
                == str(
                    fold[
                        "dataset"
                    ]
                )
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

            destination = (
                output_root
                / "within_experiment"
                / str(
                    fold[
                        "dataset"
                    ]
                )
                / (
                    f"outer-{int(fold['outer_fold_index']):03d}-"
                    f"{fold['outer_test_participant'].replace('::', '__')}"
                )
                / path_name
            )

            completion = evaluate_locked_split(
                training_frame=
                    training,
                validation_frame=
                    validation,
                test_frame=
                    test,
                candidate_features=
                    features,
                evaluation_config=
                    evaluation,
                output_directory=
                    destination,
                split_identity={
                    "protocol":
                        "within_experiment_nested_loso",
                    "dataset":
                        fold[
                            "dataset"
                        ],
                    "outer_fold_index":
                        fold[
                            "outer_fold_index"
                        ],
                    "outer_test_participant":
                        fold[
                            "outer_test_participant"
                        ],
                    "path":
                        path_name,
                },
                save_selected_model=
                    False,
            )

            execution_rows.append(
                {
                    "protocol":
                        "within_experiment",
                    "dataset_or_direction":
                        fold[
                            "dataset"
                        ],
                    "outer_test_participant":
                        fold[
                            "outer_test_participant"
                        ],
                    "path":
                        path_name,
                    "output_directory":
                        destination.as_posix(),
                    "selected_model":
                        completion[
                            "selected_model"
                        ],
                    "selected_effective_top_k":
                        completion[
                            "selected_effective_top_k"
                        ],
                    "selected_threshold":
                        completion[
                            "selected_threshold"
                        ],
                    "elapsed_seconds":
                        completion[
                            "elapsed_seconds"
                        ],
                }
            )

        for direction in plan[
            "cross_experiment_directions"
        ]:
            source_frame = frame.loc[
                frame[
                    "dataset"
                ].astype(str)
                == str(
                    direction[
                        "source"
                    ]
                )
            ].copy()

            target_frame = frame.loc[
                frame[
                    "dataset"
                ].astype(str)
                == str(
                    direction[
                        "target"
                    ]
                )
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

            destination = (
                output_root
                / "cross_experiment"
                / str(
                    direction[
                        "name"
                    ]
                )
                / path_name
            )

            completion = evaluate_locked_split(
                training_frame=
                    training,
                validation_frame=
                    validation,
                test_frame=
                    target_frame,
                candidate_features=
                    features,
                evaluation_config=
                    evaluation,
                output_directory=
                    destination,
                split_identity={
                    "protocol":
                        "cross_experiment_transfer",
                    "direction":
                        direction[
                            "name"
                        ],
                    "source":
                        direction[
                            "source"
                        ],
                    "target":
                        direction[
                            "target"
                        ],
                    "path":
                        path_name,
                },
                save_selected_model=
                    False,
            )

            execution_rows.append(
                {
                    "protocol":
                        "cross_experiment",
                    "dataset_or_direction":
                        direction[
                            "name"
                        ],
                    "outer_test_participant":
                        "",
                    "path":
                        path_name,
                    "output_directory":
                        destination.as_posix(),
                    "selected_model":
                        completion[
                            "selected_model"
                        ],
                    "selected_effective_top_k":
                        completion[
                            "selected_effective_top_k"
                        ],
                    "selected_threshold":
                        completion[
                            "selected_threshold"
                        ],
                    "elapsed_seconds":
                        completion[
                            "elapsed_seconds"
                        ],
                }
            )

    execution_table = pd.DataFrame(
        execution_rows
    )

    write_csv_atomic(
        output_root
        / "execution_registry.csv",
        execution_table,
    )

    summary = {
        "protocol_name":
            evaluation[
                "protocol_name"
            ],
        "paths_requested":
            paths,
        "completed_split_path_evaluations":
            int(
                len(
                    execution_table
                )
            ),
        "path_limit":
            path_limit,
        "fold_limit":
            fold_limit,
        "partial_run":
            bool(
                path_limit is not None
                or fold_limit is not None
            ),
        "performance_observed":
            True,
    }

    write_json_atomic(
        output_root
        / "run_summary.json",
        summary,
    )

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the leakage-safe BBBD evaluation or its synthetic smoke."
        )
    )

    mode = parser.add_mutually_exclusive_group(
        required=True
    )

    mode.add_argument(
        "--synthetic-smoke",
        action="store_true",
    )

    mode.add_argument(
        "--run-bbbd",
        action="store_true",
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
        "--plan",
        default=(
            "_research_audit/"
            "bbbd_evaluation_execution_plan_v1.json"
        ),
    )

    parser.add_argument(
        "--output-root",
        default=(
            "outputs/revision/evaluation/"
            "bbbd_v1"
        ),
    )

    parser.add_argument(
        "--path-limit",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--fold-limit",
        type=int,
        default=None,
    )

    arguments = parser.parse_args()

    if arguments.synthetic_smoke:
        result = run_synthetic_smoke()
    else:
        result = run_bbbd(
            config_path=
                arguments.config,
            feature_root=
                arguments.feature_root,
            plan_path=
                arguments.plan,
            output_root=
                arguments.output_root,
            path_limit=
                arguments.path_limit,
            fold_limit=
                arguments.fold_limit,
        )

    print(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
            default=str,
        )
    )


if __name__ == "__main__":
    main()