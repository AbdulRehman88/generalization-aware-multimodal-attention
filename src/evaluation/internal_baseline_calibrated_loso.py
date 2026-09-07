"""Baseline-calibrated nested leave-one-participant-out evaluation.

The outer participant is excluded from feature cleanup, SHAP ranking,
configuration selection, and model fitting. Only the participant's initial
known low-state baseline is used to estimate participant-specific robust
feature normalization.

Baseline-overlapping low-state windows are excluded from training, validation,
and testing. The resulting protocol is unseen-participant model evaluation
with baseline calibration. It is not calibration-free LOSO.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from src.core.config import load_revision_config
from src.evaluation.internal_calibrated_temporal_pilot import (
    build_models,
)
from src.evaluation.internal_grouped_pilot import (
    CLASS_LABELS,
)
from src.evaluation.internal_nested_loso_long_windows import (
    PROBABILITY_COLUMNS,
    causal_probability_smoothing,
    deterministic_inner_split,
    evaluate_probabilities,
    fit_classifier,
    fit_training_ranking,
    history_windows_for_duration,
    load_duration_tables,
    selected_features_for_mode,
)
from src.features.internal_xr_long_window_features import (
    model_feature_columns,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_project_path(
    value: str,
) -> Path:
    path = Path(value)

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path.resolve()


def baseline_calibrate_duration(
    frame: pd.DataFrame,
    feature_names: list[str],
    *,
    calibration_seconds: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Robustly normalize every participant using initial low baseline."""

    if calibration_seconds <= 0:
        raise ValueError(
            "calibration_seconds must be positive."
        )

    transformed = frame.copy()

    transformed[
        feature_names
    ] = transformed[
        feature_names
    ].astype(float)

    audit_rows: list[
        dict[str, Any]
    ] = []

    for participant, indices in (
        transformed.groupby(
            "participant"
        ).groups.items()
    ):
        participant_indices = np.asarray(
            list(indices),
            dtype=int,
        )

        participant_rows = transformed.loc[
            participant_indices
        ]

        baseline_mask = (
            participant_rows[
                "phase"
            ].to_numpy(dtype=int)
            == 1
        ) & (
            participant_rows[
                "end_seconds"
            ].to_numpy(dtype=float)
            <= calibration_seconds
        )

        baseline_indices = (
            participant_indices[
                baseline_mask
            ]
        )

        if len(baseline_indices) < 3:
            raise RuntimeError(
                f"{participant}: only {len(baseline_indices)} "
                "baseline windows are available."
            )

        baseline_values = transformed.loc[
            baseline_indices,
            feature_names,
        ].to_numpy(dtype=float)

        center = np.median(
            baseline_values,
            axis=0,
        )

        first_quartile = np.percentile(
            baseline_values,
            25,
            axis=0,
        )

        third_quartile = np.percentile(
            baseline_values,
            75,
            axis=0,
        )

        interquartile_range = (
            third_quartile
            - first_quartile
        )

        standard_deviation = np.std(
            baseline_values,
            axis=0,
        )

        scale = np.where(
            interquartile_range > 1e-9,
            interquartile_range,
            np.where(
                standard_deviation > 1e-9,
                standard_deviation,
                1.0,
            ),
        )

        participant_values = transformed.loc[
            participant_indices,
            feature_names,
        ].to_numpy(dtype=float)

        transformed.loc[
            participant_indices,
            feature_names,
        ] = (
            participant_values
            - center
        ) / scale

        audit_rows.append(
            {
                "participant":
                    str(participant),
                "window_seconds":
                    int(
                        participant_rows[
                            "window_seconds"
                        ].iloc[0]
                    ),
                "calibration_seconds":
                    calibration_seconds,
                "baseline_windows":
                    int(
                        len(
                            baseline_indices
                        )
                    ),
                "zero_iqr_features":
                    int(
                        (
                            interquartile_range
                            <= 1e-9
                        ).sum()
                    ),
                "unit_scale_fallback_features":
                    int(
                        (
                            (
                                interquartile_range
                                <= 1e-9
                            )
                            & (
                                standard_deviation
                                <= 1e-9
                            )
                        ).sum()
                    ),
            }
        )

    values = transformed[
        feature_names
    ].to_numpy(dtype=float)

    if not np.isfinite(
        values
    ).all():
        raise RuntimeError(
            "Baseline calibration produced non-finite values."
        )

    transformed[
        "protocol_eligible"
    ] = ~(
        (
            transformed[
                "phase"
            ].astype(int)
            == 1
        )
        & (
            transformed[
                "start_seconds"
            ].astype(float)
            < calibration_seconds
        )
    )

    return (
        transformed,
        pd.DataFrame(
            audit_rows
        ),
    )


def apply_low_class_multiplier(
    probabilities: np.ndarray,
    multiplier: float,
) -> np.ndarray:
    """Apply training-validation-selected low-class correction."""

    if multiplier <= 0:
        raise ValueError(
            "Low-class multiplier must be positive."
        )

    adjusted = np.asarray(
        probabilities,
        dtype=float,
    ).copy()

    adjusted[:, 0] *= multiplier

    adjusted /= adjusted.sum(
        axis=1,
        keepdims=True,
    )

    return adjusted


def run_baseline_calibrated_loso(
    config: dict[str, Any],
    *,
    write_outputs: bool,
) -> dict[str, Any]:
    """Run nested baseline-calibrated leave-one-participant-out."""

    training_config = config[
        "training"
    ]

    protocol = training_config[
        "baseline_calibrated_loso"
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

    calibration_seconds = int(
        protocol[
            "baseline_calibration_seconds"
        ]
    )

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

    low_multipliers = [
        float(value)
        for value in protocol[
            "low_class_probability_multipliers"
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

    raw_tables = load_duration_tables(
        feature_root,
        durations,
        modality,
    )

    tables: dict[
        int,
        pd.DataFrame,
    ] = {}

    audit_tables: list[
        pd.DataFrame
    ] = []

    for duration, frame in raw_tables.items():
        feature_names = model_feature_columns(
            frame
        )

        calibrated, audit = (
            baseline_calibrate_duration(
                frame,
                feature_names,
                calibration_seconds=(
                    calibration_seconds
                ),
            )
        )

        tables[
            duration
        ] = calibrated.loc[
            calibrated[
                "protocol_eligible"
            ]
        ].copy()

        audit[
            "duration_seconds"
        ] = duration

        audit_tables.append(
            audit
        )

    baseline_audit = pd.concat(
        audit_tables,
        ignore_index=True,
    )

    participants = sorted(
        tables[
            durations[0]
        ][
            "participant"
        ].astype(str).unique()
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

    selection_rows: list[
        dict[str, Any]
    ] = []

    validation_rows: list[
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
            f"\n===== CALIBRATED OUTER FOLD "
            f"{outer_fold:02d}/{len(participants):02d}: "
            f"TEST {test_participant} =====",
            flush=True,
        )

        fold_validation_rows: list[
            dict[str, Any]
        ] = []

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
                _,
                _,
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

            stride_values = (
                inner_validation_frame[
                    "stride_seconds"
                ]
                .astype(int)
                .unique()
            )

            if len(stride_values) != 1:
                raise RuntimeError(
                    "Inconsistent stride metadata."
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
                        + len(
                            selected_features
                        )
                    ),
                )

                for model_name in model_names:
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
                            "Unexpected model class order."
                        )

                    for low_multiplier in low_multipliers:
                        adjusted = (
                            apply_low_class_multiplier(
                                probabilities,
                                low_multiplier,
                            )
                        )

                        for requested_history in history_candidates:
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
                                    adjusted,
                                    smoothing_windows,
                                )
                            )

                            metrics = (
                                evaluate_probabilities(
                                    inner_validation_frame[
                                        "label"
                                    ].to_numpy(dtype=int),
                                    smoothed,
                                )
                            )

                            selection_score = float(
                                min(
                                    metrics["accuracy"],
                                    metrics[
                                        "balanced_accuracy"
                                    ],
                                    metrics[
                                        "macro_f1"
                                    ],
                                )
                            )

                            fold_validation_rows.append(
                                {
                                    "outer_fold":
                                        outer_fold,
                                    "outer_test_participant":
                                        test_participant,
                                    "duration_seconds":
                                        duration,
                                    "stride_seconds":
                                        stride_seconds,
                                    "top_k_mode":
                                        top_k_mode,
                                    "selected_feature_count":
                                        len(
                                            selected_features
                                        ),
                                    "model":
                                        model_name,
                                    "low_class_multiplier":
                                        low_multiplier,
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
                "low_class_multiplier",
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
                True,
            ],
            ignore_index=True,
        )

        validation_rows.extend(
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

        chosen_multiplier = float(
            chosen[
                "low_class_multiplier"
            ]
        )

        chosen_smoothing_windows = int(
            chosen[
                "smoothing_windows"
            ]
        )

        chosen_actual_history = int(
            chosen[
                "actual_history_seconds"
            ]
        )

        print(
            "Chosen:",
            {
                "duration":
                    chosen_duration,
                "top_k":
                    chosen_top_k_mode,
                "model":
                    chosen_model_name,
                "low_multiplier":
                    chosen_multiplier,
                "history_seconds":
                    chosen_actual_history,
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

        (
            retained_features,
            ranked_features,
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

        selected_features = (
            selected_features_for_mode(
                chosen_top_k_mode,
                retained_features,
                ranked_features,
            )
        )

        models = build_models(
            config,
            (
                random_seed
                + 700000
                + outer_fold
            ),
        )

        final_model = models[
            chosen_model_name
        ]

        fit_classifier(
            chosen_model_name,
            final_model,
            outer_training_frame[
                selected_features
            ],
            outer_training_frame[
                "label"
            ].to_numpy(dtype=int),
        )

        probabilities = final_model.predict_proba(
            outer_test_frame[
                selected_features
            ]
        )

        adjusted = apply_low_class_multiplier(
            probabilities,
            chosen_multiplier,
        )

        smoothed = causal_probability_smoothing(
            outer_test_frame,
            adjusted,
            chosen_smoothing_windows,
        )

        labels = outer_test_frame[
            "label"
        ].to_numpy(dtype=int)

        predictions = np.argmax(
            smoothed,
            axis=1,
        )

        metrics = evaluate_probabilities(
            labels,
            smoothed,
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
                "top_k_mode":
                    chosen_top_k_mode,
                "selected_feature_count":
                    len(
                        selected_features
                    ),
                "model":
                    chosen_model_name,
                "low_class_multiplier":
                    chosen_multiplier,
                "actual_history_seconds":
                    chosen_actual_history,
                "inner_validation_selection_score":
                    float(
                        chosen[
                            "selection_score"
                        ]
                    ),
                **metrics,
            }
        )

        selection_rows.append(
            {
                "outer_fold":
                    outer_fold,
                "test_participant":
                    test_participant,
                "duration_seconds":
                    chosen_duration,
                "top_k_mode":
                    chosen_top_k_mode,
                "model":
                    chosen_model_name,
                "low_class_multiplier":
                    chosen_multiplier,
                "smoothing_windows":
                    chosen_smoothing_windows,
                "actual_history_seconds":
                    chosen_actual_history,
                "validation_selection_score":
                    float(
                        chosen[
                            "selection_score"
                        ]
                    ),
            }
        )

        output = outer_test_frame[
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

        output = output.rename(
            columns={
                "label":
                    "true_label",
            }
        )

        output[
            "outer_fold"
        ] = outer_fold

        output[
            "predicted_label"
        ] = predictions

        for class_index, column in enumerate(
            PROBABILITY_COLUMNS
        ):
            output[
                column
            ] = smoothed[
                :,
                class_index,
            ]

        output[
            "selected_low_class_multiplier"
        ] = chosen_multiplier

        output[
            "selected_history_seconds"
        ] = chosen_actual_history

        prediction_tables.append(
            output
        )

        ranking_output = (
            final_ranking.copy()
        )

        ranking_output[
            "outer_fold"
        ] = outer_fold

        ranking_output[
            "test_participant"
        ] = test_participant

        ranking_output[
            "duration_seconds"
        ] = chosen_duration

        ranking_output[
            "used_by_final_model"
        ] = ranking_output[
            "feature"
        ].astype(str).isin(
            selected_features
        )

        ranking_tables.append(
            ranking_output
        )

        cleanup_rows.append(
            {
                "outer_fold":
                    outer_fold,
                "test_participant":
                    test_participant,
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
                        metrics["accuracy"],
                        4,
                    ),
                "balanced_accuracy":
                    round(
                        metrics[
                            "balanced_accuracy"
                        ],
                        4,
                    ),
                "macro_f1":
                    round(
                        metrics["macro_f1"],
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

    validation_results = pd.DataFrame(
        validation_rows
    )

    predictions = pd.concat(
        prediction_tables,
        ignore_index=True,
    )

    rankings = pd.concat(
        ranking_tables,
        ignore_index=True,
    )

    cleanup = pd.DataFrame(
        cleanup_rows
    )

    pooled_probabilities = predictions[
        PROBABILITY_COLUMNS
    ].to_numpy(dtype=float)

    pooled_labels = predictions[
        "true_label"
    ].to_numpy(dtype=int)

    pooled_metrics = evaluate_probabilities(
        pooled_labels,
        pooled_probabilities,
    )

    pooled_confusion = confusion_matrix(
        pooled_labels,
        predictions[
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

    phase_values = phase_probabilities[
        PROBABILITY_COLUMNS
    ].to_numpy(dtype=float)

    phase_probabilities[
        "predicted_label"
    ] = np.argmax(
        phase_values,
        axis=1,
    )

    phase_metrics = evaluate_probabilities(
        phase_probabilities[
            "true_label"
        ].to_numpy(dtype=int),
        phase_values,
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

    summary = {
        "experiment_type":
            "baseline_calibrated_nested_leave_one_participant_out",
        "classification_task":
            "three_class_low_mid_high",
        "calibration_free":
            False,
        "unseen_participant_model_training":
            True,
        "baseline_calibration_seconds":
            calibration_seconds,
        "baseline_calibration_class":
            "known_initial_low_state",
        "outer_folds":
            len(participants),
        "participant_count":
            len(participants),
        "pooled_test_decisions":
            int(
                len(
                    predictions
                )
            ),
        "pooled_metrics":
            pooled_metrics,
        "pooled_class_recalls": {
            str(index):
                float(
                    class_recalls[
                        index
                    ]
                )
            for index in range(3)
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
            int(
                len(
                    phase_probabilities
                )
            ),
        "phase_level_metrics":
            phase_metrics,
        "phase_level_confusion_matrix":
            phase_confusion.tolist(),
        "interpretation_constraint": (
            "The held-out participant is excluded from model fitting "
            "and configuration selection, but their initial known "
            "low-state baseline is used for robust normalization. "
            "This is baseline-calibrated cross-participant evaluation, "
            "not calibration-free LOSO."
        ),
    }

    if write_outputs:
        temporary = output_directory.with_name(
            output_directory.name
            + ".building"
        )

        if output_directory.exists():
            raise FileExistsError(
                "Refusing to overwrite existing output: "
                f"{output_directory}"
            )

        if temporary.exists():
            raise FileExistsError(
                f"Temporary output exists: {temporary}"
            )

        temporary.mkdir(
            parents=True,
            exist_ok=False,
        )

        fold_metrics.to_csv(
            temporary
            / "outer_fold_metrics.csv",
            index=False,
        )

        selections.to_csv(
            temporary
            / "outer_fold_selections.csv",
            index=False,
        )

        validation_results.to_csv(
            temporary
            / "inner_validation_results.csv",
            index=False,
        )

        predictions.to_csv(
            temporary
            / "outer_predictions.csv",
            index=False,
        )

        phase_probabilities.to_csv(
            temporary
            / "phase_level_predictions.csv",
            index=False,
        )

        baseline_audit.to_csv(
            temporary
            / "baseline_calibration_audit.csv",
            index=False,
        )

        rankings.to_csv(
            temporary
            / "outer_training_shap_rankings.csv",
            index=False,
        )

        cleanup.to_csv(
            temporary
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
            temporary
            / "pooled_confusion_matrix.csv"
        )

        (
            temporary
            / "baseline_calibrated_loso_summary.json"
        ).write_text(
            json.dumps(
                summary,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        temporary.replace(
            output_directory
        )

    print(
        "\n===== BASELINE-CALIBRATED LOSO RESULTS ====="
    )

    print(
        "Pooled decisions:",
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
            index:
                round(
                    float(
                        class_recalls[
                            index
                        ]
                    ),
                    4,
                )
            for index in range(3)
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
                "low_class_multiplier",
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

    print(
        "Output directory:",
        output_directory,
    )

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run baseline-calibrated nested LOSO evaluation."
        )
    )

    parser.add_argument(
        "--write",
        action="store_true",
        help="Write complete evaluation artifacts.",
    )

    arguments = parser.parse_args()

    config = load_revision_config(
        check_input_paths=True
    )

    run_baseline_calibrated_loso(
        config,
        write_outputs=arguments.write,
    )


if __name__ == "__main__":
    main()