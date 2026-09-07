"""Strict nested leave-one-participant-out evaluation.

Outer evaluation:
    One participant is completely unseen until final testing.

Inner selection:
    Two participants from the remaining cohort form a participant-disjoint
    validation set. The other participants train feature cleanup, SHAP
    ranking, and candidate models.

Selected components:
    - raw-signal window duration;
    - SHAP Top-K or all cleaned features;
    - Extra Trees or XGBoost;
    - no smoothing, approximately 60 seconds, or approximately 120 seconds
      of causal probability history.

No feature cleanup, feature selection, fitting, parameter selection, or
probability calibration uses the outer held-out participant.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight

from src.core.config import load_revision_config
from src.evaluation.internal_calibrated_temporal_pilot import (
    build_models,
)
from src.evaluation.internal_grouped_pilot import (
    CLASS_LABELS,
    compute_metrics,
    fit_training_cleanup,
    select_training_fold_features,
)
from src.features.internal_xr_long_window_features import (
    duration_directory_name,
    model_feature_columns,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROBABILITY_COLUMNS = [
    "probability_class_0",
    "probability_class_1",
    "probability_class_2",
]


def resolve_project_path(
    value: str,
) -> Path:
    """Resolve a project-relative or absolute path."""

    path = Path(value)

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path.resolve()


def deterministic_inner_split(
    participants: list[str],
    outer_test_participant: str,
    validation_count: int,
) -> tuple[list[str], list[str]]:
    """Create a deterministic participant-disjoint inner split."""

    ordered = sorted(
        str(value)
        for value in participants
    )

    if outer_test_participant not in ordered:
        raise ValueError(
            f"Outer participant not found: {outer_test_participant}"
        )

    if validation_count <= 0:
        raise ValueError(
            "validation_count must be positive."
        )

    remaining = [
        participant
        for participant in ordered
        if participant != outer_test_participant
    ]

    if len(remaining) <= validation_count:
        raise ValueError(
            "Insufficient participants for training and validation."
        )

    outer_position = ordered.index(
        outer_test_participant
    )

    validation: list[str] = []

    offset = 1

    while len(validation) < validation_count:
        candidate = ordered[
            (
                outer_position
                + offset
            )
            % len(ordered)
        ]

        offset += 1

        if (
            candidate == outer_test_participant
            or candidate in validation
        ):
            continue

        validation.append(
            candidate
        )

    training = [
        participant
        for participant in remaining
        if participant not in validation
    ]

    if set(training).intersection(
        validation
    ):
        raise RuntimeError(
            "Inner training and validation participants overlap."
        )

    if outer_test_participant in (
        training + validation
    ):
        raise RuntimeError(
            "Outer test participant entered the inner split."
        )

    return training, sorted(validation)


def history_windows_for_duration(
    requested_history_seconds: int,
    stride_seconds: int,
) -> tuple[int, int]:
    """Convert requested history duration to causal prediction windows."""

    if stride_seconds <= 0:
        raise ValueError(
            "stride_seconds must be positive."
        )

    if requested_history_seconds <= 0:
        return 1, 0

    windows = max(
        1,
        int(
            math.ceil(
                requested_history_seconds
                / stride_seconds
            )
        ),
    )

    actual_seconds = (
        windows
        * stride_seconds
    )

    return windows, actual_seconds


def causal_probability_smoothing(
    metadata: pd.DataFrame,
    probabilities: np.ndarray,
    smoothing_windows: int,
) -> np.ndarray:
    """Causally smooth probabilities without crossing phase boundaries."""

    if smoothing_windows <= 0:
        raise ValueError(
            "smoothing_windows must be positive."
        )

    work = metadata[
        [
            "participant",
            "phase",
            "long_window_index",
        ]
    ].copy().reset_index(drop=True)

    work[
        PROBABILITY_COLUMNS
    ] = np.asarray(
        probabilities,
        dtype=float,
    )

    result = np.zeros_like(
        probabilities,
        dtype=float,
    )

    grouped = work.groupby(
        [
            "participant",
            "phase",
        ],
        sort=False,
    )

    for _, indices in grouped.groups.items():
        ordered = (
            work.loc[
                list(indices)
            ]
            .sort_values(
                "long_window_index"
            )
            .index
        )

        smoothed = (
            work.loc[
                ordered,
                PROBABILITY_COLUMNS,
            ]
            .rolling(
                window=smoothing_windows,
                min_periods=1,
            )
            .mean()
            .to_numpy(dtype=float)
        )

        result[
            ordered.to_numpy(dtype=int)
        ] = smoothed

    return result


def fit_classifier(
    model_name: str,
    model: Any,
    x_train: pd.DataFrame,
    y_train: np.ndarray,
) -> None:
    """Fit a candidate tree model with class balancing."""

    if model_name == "xgboost":
        sample_weights = compute_sample_weight(
            class_weight="balanced",
            y=y_train,
        )

        model.fit(
            x_train,
            y_train,
            sample_weight=sample_weights,
        )

    else:
        model.fit(
            x_train,
            y_train,
        )


def evaluate_probabilities(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float]:
    """Compute multiclass metrics from probabilities."""

    predictions = np.argmax(
        probabilities,
        axis=1,
    )

    return compute_metrics(
        labels,
        predictions,
        probabilities,
    )


def selected_features_for_mode(
    mode: str,
    retained_features: list[str],
    ranked_features: list[str],
) -> list[str]:
    """Resolve Top-K or all cleaned training features."""

    mode_text = str(mode).strip().lower()

    if mode_text == "all":
        selected = list(
            retained_features
        )
    else:
        top_k = int(
            mode_text
        )

        selected = ranked_features[
            :min(
                top_k,
                len(ranked_features),
            )
        ]

    if not selected:
        raise RuntimeError(
            f"No features were selected for mode {mode}."
        )

    return selected


def load_duration_tables(
    feature_root: Path,
    durations: list[int],
    modality: str,
) -> dict[int, pd.DataFrame]:
    """Load and validate each configured duration table."""

    tables: dict[
        int,
        pd.DataFrame,
    ] = {}

    expected_participants: set[str] | None = None

    for duration in durations:
        directory = (
            feature_root
            / duration_directory_name(
                duration,
                0.50,
            )
        )

        path = (
            directory
            / f"{modality}_features.csv"
        )

        if not path.is_file():
            raise FileNotFoundError(
                f"Long-window feature table not found: {path}"
            )

        frame = pd.read_csv(
            path,
            low_memory=False,
        )

        required = {
            "segment_id",
            "participant",
            "phase",
            "label",
            "long_window_index",
            "window_seconds",
            "stride_seconds",
        }

        missing = required - set(
            frame.columns
        )

        if missing:
            raise RuntimeError(
                f"{path} lacks columns: {sorted(missing)}"
            )

        if frame[
            "segment_id"
        ].duplicated().any():
            raise RuntimeError(
                f"{path} contains duplicate segment identifiers."
            )

        participants = set(
            frame[
                "participant"
            ].astype(str).unique()
        )

        if expected_participants is None:
            expected_participants = participants
        elif participants != expected_participants:
            raise RuntimeError(
                "Participant sets differ across durations."
            )

        if set(
            frame[
                "label"
            ].astype(int).unique()
        ) != {
            0,
            1,
            2,
        }:
            raise RuntimeError(
                f"{path} does not contain all three classes."
            )

        observed_durations = set(
            frame[
                "window_seconds"
            ].astype(int).unique()
        )

        if observed_durations != {
            duration
        }:
            raise RuntimeError(
                f"Unexpected duration metadata in {path}: "
                f"{observed_durations}"
            )

        tables[
            duration
        ] = frame

    return tables


def fit_training_ranking(
    training_frame: pd.DataFrame,
    feature_names: list[str],
    *,
    maximum_ranked_features: int,
    random_seed: int,
    shap_max_samples: int,
    selector_parameters: dict[str, Any],
) -> tuple[
    list[str],
    list[str],
    pd.DataFrame,
    dict[str, Any],
]:
    """Fit cleanup and SHAP ranking using training participants only."""

    retained_features, cleanup = (
        fit_training_cleanup(
            training_frame[
                feature_names
            ]
        )
    )

    ranking = select_training_fold_features(
        training_frame[
            retained_features
        ],
        training_frame[
            "label"
        ].to_numpy(dtype=int),
        top_k=min(
            maximum_ranked_features,
            len(retained_features),
        ),
        random_seed=random_seed,
        maximum_shap_samples=shap_max_samples,
        selector_parameters=selector_parameters,
    )

    ranked_features = ranking[
        "feature"
    ].astype(str).tolist()

    if not ranked_features:
        raise RuntimeError(
            "SHAP ranking produced no features."
        )

    return (
        retained_features,
        ranked_features,
        ranking,
        cleanup,
    )


def run_nested_loso(
    config: dict[str, Any],
    *,
    write_outputs: bool,
) -> dict[str, Any]:
    """Run strict nested leave-one-participant-out evaluation."""

    training_config = config[
        "training"
    ]

    protocol = training_config[
        "loso_long_window"
    ]

    modality = str(
        protocol["modality"]
    )

    durations = [
        int(value)
        for value in protocol[
            "durations_seconds"
        ]
    ]

    top_k_modes = [
        str(value)
        for value in protocol[
            "top_k_values"
        ]
    ]

    model_names = [
        str(value)
        for value in protocol[
            "models"
        ]
    ]

    history_candidates = [
        int(value)
        for value in protocol[
            "probability_history_seconds"
        ]
    ]

    inner_validation_count = int(
        protocol[
            "inner_validation_participants"
        ]
    )

    shap_max_samples = int(
        protocol[
            "shap_max_samples"
        ]
    )

    random_seed = int(
        training_config[
            "random_seed"
        ]
    )

    output_directory = resolve_project_path(
        protocol[
            "output_dir"
        ]
    )

    feature_root = resolve_project_path(
        config[
            "features"
        ][
            "long_windows"
        ][
            "output_dir"
        ]
    )

    tables = load_duration_tables(
        feature_root,
        durations,
        modality,
    )

    participants = sorted(
        tables[
            durations[0]
        ][
            "participant"
        ].astype(str).unique()
    )

    if len(participants) < 4:
        raise RuntimeError(
            "Nested LOSO requires at least four participants."
        )

    maximum_numeric_top_k = max(
        int(value)
        for value in top_k_modes
        if value.lower() != "all"
    )

    selector_parameters = (
        training_config[
            "shap_selector"
        ]
    )

    fold_metric_rows: list[
        dict[str, Any]
    ] = []

    validation_result_rows: list[
        dict[str, Any]
    ] = []

    selection_rows: list[
        dict[str, Any]
    ] = []

    prediction_tables: list[
        pd.DataFrame
    ] = []

    ranking_tables: list[
        pd.DataFrame
    ] = []

    cleanup_rows: list[
        dict[str, Any]
    ] = []

    print(
        "Participants:",
        participants,
        flush=True,
    )

    for outer_fold, test_participant in enumerate(
        participants,
        start=1,
    ):
        (
            inner_training_participants,
            inner_validation_participants,
        ) = deterministic_inner_split(
            participants,
            test_participant,
            inner_validation_count,
        )

        print(
            f"\n===== OUTER FOLD {outer_fold:02d}/"
            f"{len(participants):02d}: TEST {test_participant} =====",
            flush=True,
        )

        print(
            "Inner training:",
            inner_training_participants,
            flush=True,
        )

        print(
            "Inner validation:",
            inner_validation_participants,
            flush=True,
        )

        fold_validation_rows: list[
            dict[str, Any]
        ] = []

        duration_rankings: dict[
            int,
            tuple[
                list[str],
                list[str],
                pd.DataFrame,
                dict[str, Any],
            ],
        ] = {}

        for duration in durations:
            frame = tables[
                duration
            ]

            feature_names = model_feature_columns(
                frame
            )

            inner_training_frame = frame.loc[
                frame[
                    "participant"
                ].astype(str).isin(
                    inner_training_participants
                )
            ].copy()

            inner_validation_frame = frame.loc[
                frame[
                    "participant"
                ].astype(str).isin(
                    inner_validation_participants
                )
            ].copy().reset_index(drop=True)

            (
                retained_features,
                ranked_features,
                ranking,
                cleanup,
            ) = fit_training_ranking(
                inner_training_frame,
                feature_names,
                maximum_ranked_features=(
                    maximum_numeric_top_k
                ),
                random_seed=(
                    random_seed
                    + outer_fold * 1000
                    + duration
                ),
                shap_max_samples=(
                    shap_max_samples
                ),
                selector_parameters=(
                    selector_parameters
                ),
            )

            duration_rankings[
                duration
            ] = (
                retained_features,
                ranked_features,
                ranking,
                cleanup,
            )

            stride_values = (
                inner_validation_frame[
                    "stride_seconds"
                ]
                .astype(int)
                .unique()
            )

            if len(stride_values) != 1:
                raise RuntimeError(
                    f"Duration {duration}: inconsistent stride metadata."
                )

            stride_seconds = int(
                stride_values[0]
            )

            for top_k_mode in top_k_modes:
                selected_features = (
                    selected_features_for_mode(
                        top_k_mode,
                        retained_features,
                        ranked_features,
                    )
                )

                models = build_models(
                    config,
                    (
                        random_seed
                        + outer_fold * 10000
                        + duration * 100
                        + len(selected_features)
                    ),
                )

                for model_name in model_names:
                    if model_name not in models:
                        raise RuntimeError(
                            f"Unsupported model: {model_name}"
                        )

                    model = models[
                        model_name
                    ]

                    fit_classifier(
                        model_name,
                        model,
                        inner_training_frame[
                            selected_features
                        ],
                        inner_training_frame[
                            "label"
                        ].to_numpy(dtype=int),
                    )

                    probabilities = (
                        model.predict_proba(
                            inner_validation_frame[
                                selected_features
                            ]
                        )
                    )

                    if not np.array_equal(
                        model.classes_,
                        CLASS_LABELS,
                    ):
                        raise RuntimeError(
                            f"{model_name}: unexpected class order "
                            f"{model.classes_}."
                        )

                    for requested_history in (
                        history_candidates
                    ):
                        (
                            smoothing_windows,
                            actual_history_seconds,
                        ) = history_windows_for_duration(
                            requested_history,
                            stride_seconds,
                        )

                        smoothed = (
                            causal_probability_smoothing(
                                inner_validation_frame,
                                probabilities,
                                smoothing_windows,
                            )
                        )

                        metrics = evaluate_probabilities(
                            inner_validation_frame[
                                "label"
                            ].to_numpy(dtype=int),
                            smoothed,
                        )

                        selection_score = float(
                            min(
                                metrics["accuracy"],
                                metrics[
                                    "balanced_accuracy"
                                ],
                                metrics["macro_f1"],
                            )
                        )

                        row = {
                            "outer_fold":
                                outer_fold,
                            "outer_test_participant":
                                test_participant,
                            "inner_training_participants":
                                ";".join(
                                    inner_training_participants
                                ),
                            "inner_validation_participants":
                                ";".join(
                                    inner_validation_participants
                                ),
                            "duration_seconds":
                                duration,
                            "stride_seconds":
                                stride_seconds,
                            "top_k_mode":
                                top_k_mode,
                            "selected_feature_count":
                                len(selected_features),
                            "model":
                                model_name,
                            "requested_history_seconds":
                                requested_history,
                            "smoothing_windows":
                                smoothing_windows,
                            "actual_history_seconds":
                                actual_history_seconds,
                            "selection_score":
                                selection_score,
                            **metrics,
                        }

                        fold_validation_rows.append(
                            row
                        )

        fold_validation = pd.DataFrame(
            fold_validation_rows
        ).sort_values(
            [
                "selection_score",
                "balanced_accuracy",
                "macro_f1",
                "accuracy",
                "macro_roc_auc_ovr",
                "actual_history_seconds",
                "duration_seconds",
                "selected_feature_count",
            ],
            ascending=[
                False,
                False,
                False,
                False,
                False,
                True,
                True,
                True,
            ],
            ignore_index=True,
        )

        validation_result_rows.extend(
            fold_validation.to_dict(
                orient="records"
            )
        )

        chosen = fold_validation.iloc[
            0
        ].to_dict()

        chosen_duration = int(
            chosen[
                "duration_seconds"
            ]
        )

        chosen_top_k_mode = str(
            chosen[
                "top_k_mode"
            ]
        )

        chosen_model_name = str(
            chosen["model"]
        )

        chosen_smoothing_windows = int(
            chosen[
                "smoothing_windows"
            ]
        )

        chosen_actual_history_seconds = int(
            chosen[
                "actual_history_seconds"
            ]
        )

        print(
            "Chosen configuration:",
            {
                "duration":
                    chosen_duration,
                "top_k":
                    chosen_top_k_mode,
                "model":
                    chosen_model_name,
                "history_seconds":
                    chosen_actual_history_seconds,
                "validation_score":
                    round(
                        float(
                            chosen[
                                "selection_score"
                            ]
                        ),
                        4,
                    ),
            },
            flush=True,
        )

        final_frame = tables[
            chosen_duration
        ]

        final_feature_names = model_feature_columns(
            final_frame
        )

        outer_training_participants = [
            participant
            for participant in participants
            if participant != test_participant
        ]

        outer_training_frame = final_frame.loc[
            final_frame[
                "participant"
            ].astype(str).isin(
                outer_training_participants
            )
        ].copy()

        outer_test_frame = final_frame.loc[
            final_frame[
                "participant"
            ].astype(str)
            == test_participant
        ].copy().reset_index(drop=True)

        if set(
            outer_training_frame[
                "participant"
            ].astype(str).unique()
        ).intersection(
            set(
                outer_test_frame[
                    "participant"
                ].astype(str).unique()
            )
        ):
            raise RuntimeError(
                "Outer training and test participants overlap."
            )

        (
            final_retained_features,
            final_ranked_features,
            final_ranking,
            final_cleanup,
        ) = fit_training_ranking(
            outer_training_frame,
            final_feature_names,
            maximum_ranked_features=(
                maximum_numeric_top_k
            ),
            random_seed=(
                random_seed
                + 500000
                + outer_fold
            ),
            shap_max_samples=(
                shap_max_samples
            ),
            selector_parameters=(
                selector_parameters
            ),
        )

        final_selected_features = (
            selected_features_for_mode(
                chosen_top_k_mode,
                final_retained_features,
                final_ranked_features,
            )
        )

        final_models = build_models(
            config,
            (
                random_seed
                + 700000
                + outer_fold
            ),
        )

        final_model = final_models[
            chosen_model_name
        ]

        fit_classifier(
            chosen_model_name,
            final_model,
            outer_training_frame[
                final_selected_features
            ],
            outer_training_frame[
                "label"
            ].to_numpy(dtype=int),
        )

        outer_probabilities = (
            final_model.predict_proba(
                outer_test_frame[
                    final_selected_features
                ]
            )
        )

        if not np.array_equal(
            final_model.classes_,
            CLASS_LABELS,
        ):
            raise RuntimeError(
                "Final LOSO model returned an unexpected class order."
            )

        outer_smoothed_probabilities = (
            causal_probability_smoothing(
                outer_test_frame,
                outer_probabilities,
                chosen_smoothing_windows,
            )
        )

        outer_labels = outer_test_frame[
            "label"
        ].to_numpy(dtype=int)

        outer_predictions = np.argmax(
            outer_smoothed_probabilities,
            axis=1,
        )

        outer_metrics = evaluate_probabilities(
            outer_labels,
            outer_smoothed_probabilities,
        )

        fold_metric_rows.append(
            {
                "outer_fold":
                    outer_fold,
                "test_participant":
                    test_participant,
                "test_decisions":
                    len(
                        outer_test_frame
                    ),
                "duration_seconds":
                    chosen_duration,
                "stride_seconds":
                    int(
                        outer_test_frame[
                            "stride_seconds"
                        ].iloc[0]
                    ),
                "top_k_mode":
                    chosen_top_k_mode,
                "selected_feature_count":
                    len(
                        final_selected_features
                    ),
                "model":
                    chosen_model_name,
                "actual_history_seconds":
                    chosen_actual_history_seconds,
                "inner_validation_selection_score":
                    float(
                        chosen[
                            "selection_score"
                        ]
                    ),
                **outer_metrics,
            }
        )

        selection_rows.append(
            {
                "outer_fold":
                    outer_fold,
                "test_participant":
                    test_participant,
                "inner_training_participants":
                    ";".join(
                        inner_training_participants
                    ),
                "inner_validation_participants":
                    ";".join(
                        inner_validation_participants
                    ),
                "duration_seconds":
                    chosen_duration,
                "top_k_mode":
                    chosen_top_k_mode,
                "selected_feature_count":
                    len(
                        final_selected_features
                    ),
                "model":
                    chosen_model_name,
                "smoothing_windows":
                    chosen_smoothing_windows,
                "actual_history_seconds":
                    chosen_actual_history_seconds,
                "validation_selection_score":
                    float(
                        chosen[
                            "selection_score"
                        ]
                    ),
            }
        )

        prediction_output = outer_test_frame[
            [
                "segment_id",
                "participant",
                "phase",
                "label",
                "long_window_index",
                "start_seconds",
                "end_seconds",
                "window_seconds",
                "stride_seconds",
            ]
        ].copy()

        prediction_output = prediction_output.rename(
            columns={
                "label": "true_label",
            }
        )

        prediction_output[
            "outer_fold"
        ] = outer_fold

        prediction_output[
            "predicted_label"
        ] = outer_predictions

        for class_index, column in enumerate(
            PROBABILITY_COLUMNS
        ):
            prediction_output[
                column
            ] = outer_smoothed_probabilities[
                :,
                class_index,
            ]

        prediction_output[
            "selected_duration_seconds"
        ] = chosen_duration

        prediction_output[
            "selected_top_k_mode"
        ] = chosen_top_k_mode

        prediction_output[
            "selected_model"
        ] = chosen_model_name

        prediction_output[
            "selected_history_seconds"
        ] = chosen_actual_history_seconds

        prediction_tables.append(
            prediction_output
        )

        final_ranking_output = (
            final_ranking.copy()
        )

        final_ranking_output[
            "outer_fold"
        ] = outer_fold

        final_ranking_output[
            "test_participant"
        ] = test_participant

        final_ranking_output[
            "duration_seconds"
        ] = chosen_duration

        final_ranking_output[
            "selected_top_k_mode"
        ] = chosen_top_k_mode

        final_ranking_output[
            "used_by_final_model"
        ] = final_ranking_output[
            "feature"
        ].astype(str).isin(
            final_selected_features
        )

        ranking_tables.append(
            final_ranking_output
        )

        cleanup_rows.append(
            {
                "outer_fold":
                    outer_fold,
                "test_participant":
                    test_participant,
                "duration_seconds":
                    chosen_duration,
                **{
                    key: (
                        ";".join(value)
                        if isinstance(
                            value,
                            list,
                        )
                        else value
                    )
                    for key, value
                    in final_cleanup.items()
                },
            }
        )

        print(
            "Outer test:",
            {
                "accuracy":
                    round(
                        outer_metrics[
                            "accuracy"
                        ],
                        4,
                    ),
                "balanced_accuracy":
                    round(
                        outer_metrics[
                            "balanced_accuracy"
                        ],
                        4,
                    ),
                "macro_f1":
                    round(
                        outer_metrics[
                            "macro_f1"
                        ],
                        4,
                    ),
            },
            flush=True,
        )

    fold_metrics = pd.DataFrame(
        fold_metric_rows
    ).sort_values(
        "outer_fold",
        ignore_index=True,
    )

    selections = pd.DataFrame(
        selection_rows
    ).sort_values(
        "outer_fold",
        ignore_index=True,
    )

    predictions = pd.concat(
        prediction_tables,
        ignore_index=True,
    )

    rankings = pd.concat(
        ranking_tables,
        ignore_index=True,
    )

    cleanup_table = pd.DataFrame(
        cleanup_rows
    )

    validation_results = pd.DataFrame(
        validation_result_rows
    )

    if predictions[
        "segment_id"
    ].duplicated().any():
        raise RuntimeError(
            "Pooled LOSO predictions contain duplicate segment IDs."
        )

    if predictions[
        "participant"
    ].nunique() != len(
        participants
    ):
        raise RuntimeError(
            "Pooled LOSO predictions do not cover all participants."
        )

    pooled_metrics = evaluate_probabilities(
        predictions[
            "true_label"
        ].to_numpy(dtype=int),
        predictions[
            PROBABILITY_COLUMNS
        ].to_numpy(dtype=float),
    )

    pooled_confusion = confusion_matrix(
        predictions[
            "true_label"
        ],
        predictions[
            "predicted_label"
        ],
        labels=CLASS_LABELS,
    )

    phase_probabilities = (
        predictions.groupby(
            [
                "participant",
                "phase",
                "true_label",
            ],
            as_index=False,
        )[
            PROBABILITY_COLUMNS
        ]
        .mean()
    )

    phase_probability_values = (
        phase_probabilities[
            PROBABILITY_COLUMNS
        ].to_numpy(dtype=float)
    )

    phase_probabilities[
        "predicted_label"
    ] = np.argmax(
        phase_probability_values,
        axis=1,
    )

    phase_metrics = evaluate_probabilities(
        phase_probabilities[
            "true_label"
        ].to_numpy(dtype=int),
        phase_probability_values,
    )

    phase_confusion = confusion_matrix(
        phase_probabilities[
            "true_label"
        ],
        phase_probabilities[
            "predicted_label"
        ],
        labels=CLASS_LABELS,
    )

    class_recalls = (
        np.diag(
            pooled_confusion
        )
        / np.maximum(
            pooled_confusion.sum(
                axis=1
            ),
            1,
        )
    )

    summary = {
        "experiment_type":
            "strict_nested_leave_one_participant_out",
        "classification_task":
            "three_class_low_mid_high",
        "calibration_free":
            True,
        "unseen_participant_generalization":
            True,
        "outer_folds":
            len(participants),
        "participant_count":
            len(participants),
        "participants":
            participants,
        "inner_validation_participants_per_fold":
            inner_validation_count,
        "candidate_durations_seconds":
            durations,
        "candidate_top_k_modes":
            top_k_modes,
        "candidate_models":
            model_names,
        "candidate_probability_history_seconds":
            history_candidates,
        "pooled_test_decisions":
            int(len(predictions)),
        "pooled_metrics":
            pooled_metrics,
        "pooled_class_recalls": {
            str(class_index):
                float(
                    class_recalls[
                        class_index
                    ]
                )
            for class_index in range(3)
        },
        "pooled_confusion_matrix":
            pooled_confusion.tolist(),
        "fold_metric_mean": {
            metric:
                float(
                    fold_metrics[
                        metric
                    ].mean()
                )
            for metric in [
                "accuracy",
                "balanced_accuracy",
                "macro_f1",
                "macro_roc_auc_ovr",
                "macro_pr_auc",
            ]
        },
        "fold_metric_standard_deviation": {
            metric:
                float(
                    fold_metrics[
                        metric
                    ].std(
                        ddof=1
                    )
                )
            for metric in [
                "accuracy",
                "balanced_accuracy",
                "macro_f1",
                "macro_roc_auc_ovr",
                "macro_pr_auc",
            ]
        },
        "phase_level_decisions":
            int(len(phase_probabilities)),
        "phase_level_metrics":
            phase_metrics,
        "phase_level_confusion_matrix":
            phase_confusion.tolist(),
        "selection_frequency": {
            "duration_seconds":
                selections[
                    "duration_seconds"
                ].value_counts().sort_index().to_dict(),
            "top_k_mode":
                selections[
                    "top_k_mode"
                ].value_counts().to_dict(),
            "model":
                selections[
                    "model"
                ].value_counts().to_dict(),
            "actual_history_seconds":
                selections[
                    "actual_history_seconds"
                ].value_counts().sort_index().to_dict(),
        },
        "interpretation_constraint": (
            "Every outer prediction concerns a participant excluded "
            "from feature cleanup, SHAP ranking, model selection, "
            "model fitting, and probability-parameter selection. "
            "Overlapping predictions are temporally correlated and "
            "must not be treated as independent observations."
        ),
    }

    if write_outputs:
        temporary_directory = (
            output_directory.with_name(
                output_directory.name
                + ".building"
            )
        )

        if output_directory.exists():
            raise FileExistsError(
                "Refusing to overwrite existing LOSO output: "
                f"{output_directory}"
            )

        if temporary_directory.exists():
            raise FileExistsError(
                "Temporary LOSO output already exists: "
                f"{temporary_directory}"
            )

        temporary_directory.mkdir(
            parents=True,
            exist_ok=False,
        )

        fold_metrics.to_csv(
            temporary_directory
            / "outer_fold_metrics.csv",
            index=False,
        )

        selections.to_csv(
            temporary_directory
            / "outer_fold_selections.csv",
            index=False,
        )

        validation_results.to_csv(
            temporary_directory
            / "inner_validation_results.csv",
            index=False,
        )

        predictions.to_csv(
            temporary_directory
            / "outer_loso_predictions.csv",
            index=False,
        )

        phase_probabilities.to_csv(
            temporary_directory
            / "phase_level_predictions.csv",
            index=False,
        )

        rankings.to_csv(
            temporary_directory
            / "outer_training_shap_rankings.csv",
            index=False,
        )

        cleanup_table.to_csv(
            temporary_directory
            / "outer_training_cleanup.csv",
            index=False,
        )

        pd.DataFrame(
            pooled_confusion,
            index=[
                "true_0",
                "true_1",
                "true_2",
            ],
            columns=[
                "predicted_0",
                "predicted_1",
                "predicted_2",
            ],
        ).to_csv(
            temporary_directory
            / "pooled_confusion_matrix.csv"
        )

        pd.DataFrame(
            phase_confusion,
            index=[
                "true_0",
                "true_1",
                "true_2",
            ],
            columns=[
                "predicted_0",
                "predicted_1",
                "predicted_2",
            ],
        ).to_csv(
            temporary_directory
            / "phase_level_confusion_matrix.csv"
        )

        (
            temporary_directory
            / "nested_loso_summary.json"
        ).write_text(
            json.dumps(
                summary,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        temporary_directory.replace(
            output_directory
        )

    print(
        "\n===== STRICT NESTED LOSO RESULTS ====="
    )

    print(
        "Pooled test decisions:",
        len(predictions),
    )

    print(
        "Accuracy:",
        f"{pooled_metrics['accuracy']:.4f}",
    )

    print(
        "Balanced accuracy:",
        f"{pooled_metrics['balanced_accuracy']:.4f}",
    )

    print(
        "Macro-F1:",
        f"{pooled_metrics['macro_f1']:.4f}",
    )

    print(
        "Macro ROC-AUC:",
        f"{pooled_metrics['macro_roc_auc_ovr']:.4f}",
    )

    print(
        "Macro PR-AUC:",
        f"{pooled_metrics['macro_pr_auc']:.4f}",
    )

    print(
        "Class recalls:",
        {
            class_index:
                round(
                    float(
                        class_recalls[
                            class_index
                        ]
                    ),
                    4,
                )
            for class_index in range(3)
        },
    )

    print(
        "Confusion matrix:"
    )

    print(
        pooled_confusion
    )

    print(
        "\n===== OUTER-FOLD RESULTS ====="
    )

    print(
        fold_metrics[
            [
                "test_participant",
                "duration_seconds",
                "top_k_mode",
                "model",
                "actual_history_seconds",
                "accuracy",
                "balanced_accuracy",
                "macro_f1",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\n===== PHASE-LEVEL RESULTS ====="
    )

    print(
        "Decisions:",
        len(phase_probabilities),
    )

    print(
        "Accuracy:",
        f"{phase_metrics['accuracy']:.4f}",
    )

    print(
        "Balanced accuracy:",
        f"{phase_metrics['balanced_accuracy']:.4f}",
    )

    print(
        "Macro-F1:",
        f"{phase_metrics['macro_f1']:.4f}",
    )

    if write_outputs:
        print(
            "Output directory:",
            output_directory,
        )

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run strict nested long-window LOSO evaluation."
        )
    )

    parser.add_argument(
        "--write",
        action="store_true",
        help="Write complete LOSO artifacts.",
    )

    arguments = parser.parse_args()

    config = load_revision_config(
        check_input_paths=True
    )

    run_nested_loso(
        config,
        write_outputs=arguments.write,
    )


if __name__ == "__main__":
    main()