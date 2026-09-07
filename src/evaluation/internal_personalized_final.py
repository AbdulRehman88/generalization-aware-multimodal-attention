"""Validation-selected personalized three-class attention evaluation.

The protocol has five nonoverlapping temporal roles within every participant
and experimental phase:

1. Calibration
2. First guard interval
3. Validation
4. Second guard interval
5. Untouched final test

The validation interval selects:

- Top-K feature count
- classifier
- participant-prototype temperature
- model/prototype probability blending
- low-class probability calibration
- causal temporal smoothing duration

The selected configuration is then refitted using calibration plus validation
labels and evaluated once on the untouched final test interval.

This is a subject-calibrated personalized protocol. It is not calibration-free
or unseen-participant generalization.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight

from src.core.config import load_revision_config
from src.evaluation.internal_calibrated_temporal_pilot import (
    build_causal_context,
    build_models,
    robust_baseline_normalize,
)
from src.evaluation.internal_grouped_pilot import (
    CLASS_LABELS,
    compute_metrics,
    feature_columns,
    fit_training_cleanup,
    select_training_fold_features,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_project_path(
    value: str,
) -> Path:
    path = Path(value)

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path.resolve()


def assign_final_roles(
    context: pd.DataFrame,
    *,
    calibration_windows: int,
    first_guard_windows: int,
    validation_windows: int,
    second_guard_windows: int,
) -> pd.Series:
    """Assign calibration, validation, guard, and untouched test roles."""

    calibration_end = calibration_windows

    first_guard_end = (
        calibration_end
        + first_guard_windows
    )

    validation_end = (
        first_guard_end
        + validation_windows
    )

    second_guard_end = (
        validation_end
        + second_guard_windows
    )

    indices = context[
        "window_index"
    ].to_numpy(dtype=int)

    roles = np.select(
        [
            indices < calibration_end,
            indices < first_guard_end,
            indices < validation_end,
            indices < second_guard_end,
        ],
        [
            "calibration",
            "guard_1",
            "validation",
            "guard_2",
        ],
        default="test",
    )

    return pd.Series(
        roles,
        index=context.index,
        dtype="object",
    )


@dataclass
class PrototypeSpace:
    """Participant-specific class-prototype representation."""

    center: np.ndarray
    scale: np.ndarray
    centroids: dict[tuple[str, int], np.ndarray]


def fit_prototype_space(
    adaptation_frame: pd.DataFrame,
    selected_features: list[str],
) -> PrototypeSpace:
    """Fit participant-specific class prototypes without test data."""

    values = adaptation_frame[
        selected_features
    ].to_numpy(dtype=float)

    center = np.median(
        values,
        axis=0,
    )

    first_quartile = np.percentile(
        values,
        25,
        axis=0,
    )

    third_quartile = np.percentile(
        values,
        75,
        axis=0,
    )

    interquartile_range = (
        third_quartile
        - first_quartile
    )

    standard_deviation = np.std(
        values,
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

    standardized = (
        values
        - center
    ) / scale

    participants = adaptation_frame[
        "participant"
    ].astype(str).to_numpy()

    labels = adaptation_frame[
        "label"
    ].to_numpy(dtype=int)

    centroids: dict[
        tuple[str, int],
        np.ndarray,
    ] = {}

    for participant in np.unique(
        participants
    ):
        for label in CLASS_LABELS:
            mask = (
                (participants == participant)
                & (labels == label)
            )

            if not mask.any():
                raise RuntimeError(
                    f"Missing prototype rows for "
                    f"{participant}, class {label}."
                )

            centroids[
                (
                    str(participant),
                    int(label),
                )
            ] = np.mean(
                standardized[mask],
                axis=0,
            )

    return PrototypeSpace(
        center=center,
        scale=scale,
        centroids=centroids,
    )


def compute_prototype_distances(
    prototype_space: PrototypeSpace,
    target_frame: pd.DataFrame,
    selected_features: list[str],
) -> np.ndarray:
    """Compute participant-specific distances to all class prototypes."""

    values = target_frame[
        selected_features
    ].to_numpy(dtype=float)

    standardized = (
        values
        - prototype_space.center
    ) / prototype_space.scale

    participants = target_frame[
        "participant"
    ].astype(str).to_numpy()

    distances = np.zeros(
        (
            len(target_frame),
            len(CLASS_LABELS),
        ),
        dtype=float,
    )

    for row_index, participant in enumerate(
        participants
    ):
        for class_index, label in enumerate(
            CLASS_LABELS
        ):
            key = (
                str(participant),
                int(label),
            )

            if key not in prototype_space.centroids:
                raise RuntimeError(
                    f"Prototype not found: {key}"
                )

            difference = (
                standardized[row_index]
                - prototype_space.centroids[key]
            )

            distances[
                row_index,
                class_index,
            ] = float(
                np.mean(
                    difference ** 2
                )
            )

    if not np.isfinite(distances).all():
        raise RuntimeError(
            "Prototype distances contain non-finite values."
        )

    return distances


def distances_to_probabilities(
    distances: np.ndarray,
    temperature: float,
) -> np.ndarray:
    """Convert class distances to stable probability estimates."""

    if temperature <= 0:
        raise ValueError(
            "Prototype temperature must be positive."
        )

    logits = (
        -np.asarray(
            distances,
            dtype=float,
        )
        / temperature
    )

    logits = (
        logits
        - np.max(
            logits,
            axis=1,
            keepdims=True,
        )
    )

    exponentials = np.exp(
        logits
    )

    probabilities = (
        exponentials
        / np.sum(
            exponentials,
            axis=1,
            keepdims=True,
        )
    )

    return probabilities


def blend_probabilities(
    model_probabilities: np.ndarray,
    prototype_probabilities: np.ndarray,
    model_weight: float,
) -> np.ndarray:
    """Blend global-model and personalized-prototype probabilities."""

    if not 0.0 <= model_weight <= 1.0:
        raise ValueError(
            "Model probability weight must be in [0, 1]."
        )

    blended = (
        model_weight
        * model_probabilities
        + (
            1.0
            - model_weight
        )
        * prototype_probabilities
    )

    blended = (
        blended
        / np.sum(
            blended,
            axis=1,
            keepdims=True,
        )
    )

    return blended


def apply_low_class_multiplier(
    probabilities: np.ndarray,
    multiplier: float,
) -> np.ndarray:
    """Apply validation-selected prior calibration to the low class."""

    if multiplier <= 0:
        raise ValueError(
            "Low-class multiplier must be positive."
        )

    adjusted = np.asarray(
        probabilities,
        dtype=float,
    ).copy()

    adjusted[:, 0] *= multiplier

    adjusted /= np.sum(
        adjusted,
        axis=1,
        keepdims=True,
    )

    return adjusted


def causal_smooth_probabilities(
    metadata: pd.DataFrame,
    probabilities: np.ndarray,
    smoothing_windows: int,
) -> np.ndarray:
    """Causally average probabilities within participant and phase."""

    if smoothing_windows <= 0:
        raise ValueError(
            "Smoothing window count must be positive."
        )

    work = metadata[
        [
            "participant",
            "phase",
            "window_index",
        ]
    ].copy()

    probability_columns = [
        "probability_0",
        "probability_1",
        "probability_2",
    ]

    work[
        probability_columns
    ] = probabilities

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
        ordered_indices = (
            work.loc[
                list(indices)
            ]
            .sort_values(
                "window_index"
            )
            .index
        )

        smoothed = (
            work.loc[
                ordered_indices,
                probability_columns,
            ]
            .rolling(
                window=smoothing_windows,
                min_periods=1,
            )
            .mean()
            .to_numpy(dtype=float)
        )

        result[
            ordered_indices.to_numpy(
                dtype=int
            )
        ] = smoothed

    return result


def evaluate_probability_rows(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float]:
    """Compute metrics from probability estimates."""

    predictions = np.argmax(
        probabilities,
        axis=1,
    )

    return compute_metrics(
        labels,
        predictions,
        probabilities,
    )


def fit_model(
    model_name: str,
    model: Any,
    x_train: pd.DataFrame,
    y_train: np.ndarray,
) -> None:
    """Fit one classifier with appropriate class balancing."""

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


def rank_training_features(
    adaptation_frame: pd.DataFrame,
    temporal_features: list[str],
    config: dict[str, Any],
    maximum_top_k: int,
    random_seed: int,
    shap_max_samples: int,
) -> tuple[
    list[str],
    pd.DataFrame,
    dict[str, Any],
]:
    """Fit cleanup and one SHAP ranking on adaptation-only rows."""

    retained_features, cleanup = (
        fit_training_cleanup(
            adaptation_frame[
                temporal_features
            ]
        )
    )

    ranking = select_training_fold_features(
        adaptation_frame[
            retained_features
        ],
        adaptation_frame[
            "label"
        ].to_numpy(dtype=int),
        top_k=min(
            maximum_top_k,
            len(retained_features),
        ),
        random_seed=random_seed,
        maximum_shap_samples=shap_max_samples,
        selector_parameters=config[
            "training"
        ][
            "shap_selector"
        ],
    )

    ranked_features = ranking[
        "feature"
    ].tolist()

    return (
        ranked_features,
        ranking,
        cleanup,
    )


def run_personalized_final(
    config: dict[str, Any],
    *,
    write_outputs: bool,
) -> dict[str, Any]:
    """Select on validation and evaluate once on untouched test data."""

    training_config = config[
        "training"
    ]

    protocol = training_config[
        "personalized_final"
    ]

    modality = str(
        protocol["modality"]
    )

    calibration_windows = int(
        protocol[
            "calibration_windows_per_phase"
        ]
    )

    first_guard_windows = int(
        protocol[
            "first_guard_windows"
        ]
    )

    validation_windows = int(
        protocol[
            "validation_windows_per_phase"
        ]
    )

    second_guard_windows = int(
        protocol[
            "second_guard_windows"
        ]
    )

    context_windows = int(
        protocol["context_windows"]
    )

    top_k_values = [
        int(value)
        for value in protocol[
            "top_k_values"
        ]
    ]

    temperatures = [
        float(value)
        for value in protocol[
            "prototype_temperatures"
        ]
    ]

    model_weights = [
        float(value)
        for value in protocol[
            "model_probability_weights"
        ]
    ]

    low_multipliers = [
        float(value)
        for value in protocol[
            "low_class_probability_multipliers"
        ]
    ]

    smoothing_values = [
        int(value)
        for value in protocol[
            "probability_smoothing_windows"
        ]
    ]

    shap_max_samples = int(
        protocol["shap_max_samples"]
    )

    refit_with_validation = bool(
        protocol["refit_with_validation"]
    )

    random_seed = int(
        training_config[
            "random_seed"
        ]
    )

    feature_directory = resolve_project_path(
        config[
            "features"
        ][
            "output_dir"
        ]
    )

    feature_path = (
        feature_directory
        / f"{modality}_features.csv"
    )

    output_directory = resolve_project_path(
        protocol["output_dir"]
    )

    if not feature_path.is_file():
        raise FileNotFoundError(
            f"Feature matrix not found: {feature_path}"
        )

    frame = pd.read_csv(
        feature_path,
        low_memory=False,
    )

    base_features = feature_columns(
        frame
    )

    normalized, baseline_audit = (
        robust_baseline_normalize(
            frame,
            base_features,
            calibration_windows,
        )
    )

    print(
        "Building 20-second causal contexts...",
        flush=True,
    )

    context = build_causal_context(
        normalized,
        base_features,
        context_windows,
    ).reset_index(drop=True)

    context["role"] = assign_final_roles(
        context,
        calibration_windows=calibration_windows,
        first_guard_windows=first_guard_windows,
        validation_windows=validation_windows,
        second_guard_windows=second_guard_windows,
    )

    role_counts = (
        context.groupby(
            [
                "participant",
                "phase",
                "role",
            ]
        )
        .size()
        .rename("rows")
        .reset_index()
    )

    temporal_features = [
        column
        for column in context.columns
        if column.startswith(
            (
                "context_mean__",
                "context_std__",
                "context_slope__",
            )
        )
    ]

    calibration_frame = context.loc[
        context["role"]
        == "calibration"
    ].copy()

    validation_inference = context.loc[
        context["role"].isin(
            [
                "guard_1",
                "validation",
            ]
        )
    ].copy().reset_index(drop=True)

    validation_mask = (
        validation_inference[
            "role"
        ].to_numpy()
        == "validation"
    )

    if (
        calibration_frame.empty
        or not validation_mask.any()
    ):
        raise RuntimeError(
            "Calibration or validation partition is empty."
        )

    maximum_top_k = max(
        top_k_values
    )

    (
        validation_ranked_features,
        validation_ranking,
        validation_cleanup,
    ) = rank_training_features(
        calibration_frame,
        temporal_features,
        config,
        maximum_top_k,
        random_seed,
        shap_max_samples,
    )

    validation_rows: list[
        dict[str, Any]
    ] = []

    print(
        "Running validation-only configuration search...",
        flush=True,
    )

    for top_k in top_k_values:
        selected_features = (
            validation_ranked_features[
                :min(
                    top_k,
                    len(
                        validation_ranked_features
                    ),
                )
            ]
        )

        models = build_models(
            config,
            random_seed + top_k,
        )

        prototype_space = fit_prototype_space(
            calibration_frame,
            selected_features,
        )

        validation_distances = (
            compute_prototype_distances(
                prototype_space,
                validation_inference,
                selected_features,
            )
        )

        for model_name, model in models.items():
            fit_model(
                model_name,
                model,
                calibration_frame[
                    selected_features
                ],
                calibration_frame[
                    "label"
                ].to_numpy(dtype=int),
            )

            model_probabilities = (
                model.predict_proba(
                    validation_inference[
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

            for temperature in temperatures:
                prototype_probabilities = (
                    distances_to_probabilities(
                        validation_distances,
                        temperature,
                    )
                )

                for model_weight in model_weights:
                    blended = blend_probabilities(
                        model_probabilities,
                        prototype_probabilities,
                        model_weight,
                    )

                    for low_multiplier in low_multipliers:
                        adjusted = (
                            apply_low_class_multiplier(
                                blended,
                                low_multiplier,
                            )
                        )

                        for smoothing in smoothing_values:
                            smoothed = (
                                causal_smooth_probabilities(
                                    validation_inference,
                                    adjusted,
                                    smoothing,
                                )
                            )

                            validation_probabilities = (
                                smoothed[
                                    validation_mask
                                ]
                            )

                            validation_labels = (
                                validation_inference.loc[
                                    validation_mask,
                                    "label",
                                ].to_numpy(dtype=int)
                            )

                            metrics = (
                                evaluate_probability_rows(
                                    validation_labels,
                                    validation_probabilities,
                                )
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

                            validation_rows.append(
                                {
                                    "top_k":
                                        len(
                                            selected_features
                                        ),
                                    "model":
                                        model_name,
                                    "prototype_temperature":
                                        temperature,
                                    "model_probability_weight":
                                        model_weight,
                                    "low_class_multiplier":
                                        low_multiplier,
                                    "smoothing_windows":
                                        smoothing,
                                    "probability_history_seconds":
                                        smoothing * 4,
                                    "selection_score":
                                        selection_score,
                                    **metrics,
                                }
                            )

    validation_results = pd.DataFrame(
        validation_rows
    ).sort_values(
        [
            "selection_score",
            "balanced_accuracy",
            "macro_f1",
            "accuracy",
            "macro_roc_auc_ovr",
        ],
        ascending=False,
        ignore_index=True,
    )

    chosen = validation_results.iloc[
        0
    ].to_dict()

    print(
        "Selected validation configuration:",
        json.dumps(
            chosen,
            indent=2,
        ),
        flush=True,
    )

    if refit_with_validation:
        adaptation_frame = context.loc[
            context["role"].isin(
                [
                    "calibration",
                    "validation",
                ]
            )
        ].copy()

        adaptation_description = (
            "calibration_plus_validation"
        )
    else:
        adaptation_frame = (
            calibration_frame.copy()
        )

        adaptation_description = (
            "calibration_only"
        )

    test_inference = context.loc[
        context["role"].isin(
            [
                "guard_2",
                "test",
            ]
        )
    ].copy().reset_index(drop=True)

    test_mask = (
        test_inference[
            "role"
        ].to_numpy()
        == "test"
    )

    if adaptation_frame.empty or not test_mask.any():
        raise RuntimeError(
            "Final adaptation or test partition is empty."
        )

    (
        final_ranked_features,
        final_ranking,
        final_cleanup,
    ) = rank_training_features(
        adaptation_frame,
        temporal_features,
        config,
        int(chosen["top_k"]),
        random_seed + 10000,
        shap_max_samples,
    )

    selected_features = (
        final_ranked_features[
            :int(chosen["top_k"])
        ]
    )

    final_models = build_models(
        config,
        random_seed + 20000,
    )

    chosen_model_name = str(
        chosen["model"]
    )

    final_model = final_models[
        chosen_model_name
    ]

    fit_model(
        chosen_model_name,
        final_model,
        adaptation_frame[
            selected_features
        ],
        adaptation_frame[
            "label"
        ].to_numpy(dtype=int),
    )

    model_probabilities = (
        final_model.predict_proba(
            test_inference[
                selected_features
            ]
        )
    )

    if not np.array_equal(
        final_model.classes_,
        CLASS_LABELS,
    ):
        raise RuntimeError(
            "Final model class order is incorrect."
        )

    final_prototype_space = (
        fit_prototype_space(
            adaptation_frame,
            selected_features,
        )
    )

    test_distances = (
        compute_prototype_distances(
            final_prototype_space,
            test_inference,
            selected_features,
        )
    )

    prototype_probabilities = (
        distances_to_probabilities(
            test_distances,
            float(
                chosen[
                    "prototype_temperature"
                ]
            ),
        )
    )

    blended_probabilities = (
        blend_probabilities(
            model_probabilities,
            prototype_probabilities,
            float(
                chosen[
                    "model_probability_weight"
                ]
            ),
        )
    )

    adjusted_probabilities = (
        apply_low_class_multiplier(
            blended_probabilities,
            float(
                chosen[
                    "low_class_multiplier"
                ]
            ),
        )
    )

    smoothed_probabilities = (
        causal_smooth_probabilities(
            test_inference,
            adjusted_probabilities,
            int(
                chosen[
                    "smoothing_windows"
                ]
            ),
        )
    )

    final_test_probabilities = (
        smoothed_probabilities[
            test_mask
        ]
    )

    final_test_frame = (
        test_inference.loc[
            test_mask
        ]
        .copy()
        .reset_index(drop=True)
    )

    final_test_labels = (
        final_test_frame[
            "label"
        ].to_numpy(dtype=int)
    )

    final_test_predictions = np.argmax(
        final_test_probabilities,
        axis=1,
    )

    final_test_metrics = (
        evaluate_probability_rows(
            final_test_labels,
            final_test_probabilities,
        )
    )

    final_confusion = confusion_matrix(
        final_test_labels,
        final_test_predictions,
        labels=CLASS_LABELS,
    )

    prediction_output = final_test_frame[
        [
            "segment_id",
            "participant",
            "phase",
            "label",
            "window_index",
        ]
    ].copy()

    prediction_output = prediction_output.rename(
        columns={
            "label": "true_label",
        }
    )

    prediction_output[
        "predicted_label"
    ] = final_test_predictions

    prediction_output[
        "probability_class_0"
    ] = final_test_probabilities[:, 0]

    prediction_output[
        "probability_class_1"
    ] = final_test_probabilities[:, 1]

    prediction_output[
        "probability_class_2"
    ] = final_test_probabilities[:, 2]

    phase_probabilities = (
        prediction_output.groupby(
            [
                "participant",
                "phase",
                "true_label",
            ],
            as_index=False,
        )[
            [
                "probability_class_0",
                "probability_class_1",
                "probability_class_2",
            ]
        ]
        .mean()
    )

    phase_probability_values = (
        phase_probabilities[
            [
                "probability_class_0",
                "probability_class_1",
                "probability_class_2",
            ]
        ].to_numpy(dtype=float)
    )

    phase_metrics = evaluate_probability_rows(
        phase_probabilities[
            "true_label"
        ].to_numpy(dtype=int),
        phase_probability_values,
    )

    phase_probabilities[
        "predicted_label"
    ] = np.argmax(
        phase_probability_values,
        axis=1,
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
        np.diag(final_confusion)
        / np.maximum(
            final_confusion.sum(axis=1),
            1,
        )
    )

    summary = {
        "experiment_type":
            "validation_selected_personalized_multiclass_final",
        "classification_task":
            "three_class_low_mid_high",
        "protocol_scope":
            "within_cohort_subject_calibrated",
        "calibration_free":
            False,
        "unseen_participant_generalization":
            False,
        "modality":
            modality,
        "participant_count":
            int(
                context[
                    "participant"
                ].nunique()
            ),
        "participants":
            sorted(
                context[
                    "participant"
                ].astype(str).unique().tolist()
            ),
        "context_windows":
            context_windows,
        "context_seconds":
            context_windows * 4,
        "calibration_windows_per_phase":
            calibration_windows,
        "calibration_seconds_per_phase":
            calibration_windows * 4,
        "validation_windows_per_phase":
            validation_windows,
        "validation_seconds_per_phase":
            validation_windows * 4,
        "first_guard_seconds":
            first_guard_windows * 4,
        "second_guard_seconds":
            second_guard_windows * 4,
        "final_adaptation_scope":
            adaptation_description,
        "validation_configuration":
            chosen,
        "final_selected_feature_count":
            len(selected_features),
        "final_selected_features":
            selected_features,
        "untouched_test_decisions":
            int(len(final_test_frame)),
        "untouched_test_metrics":
            final_test_metrics,
        "untouched_test_class_recalls": {
            str(class_index):
                float(class_recalls[class_index])
            for class_index in range(3)
        },
        "untouched_test_confusion_matrix":
            final_confusion.tolist(),
        "phase_level_decisions":
            int(len(phase_probabilities)),
        "phase_level_metrics":
            phase_metrics,
        "phase_level_confusion_matrix":
            phase_confusion.tolist(),
        "interpretation_constraint": (
            "The final result uses labeled participant-specific "
            "adaptation and must not be described as calibration-free "
            "cross-participant performance."
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
                "Refusing to overwrite final output: "
                f"{output_directory}"
            )

        if temporary_directory.exists():
            raise FileExistsError(
                "Temporary output directory exists: "
                f"{temporary_directory}"
            )

        temporary_directory.mkdir(
            parents=True,
            exist_ok=False,
        )

        validation_results.to_csv(
            temporary_directory
            / "validation_configuration_search.csv",
            index=False,
        )

        validation_results.head(
            25
        ).to_csv(
            temporary_directory
            / "validation_top25.csv",
            index=False,
        )

        validation_ranking.to_csv(
            temporary_directory
            / "calibration_shap_ranking.csv",
            index=False,
        )

        final_ranking.to_csv(
            temporary_directory
            / "final_adaptation_shap_ranking.csv",
            index=False,
        )

        pd.DataFrame(
            {
                "feature":
                    selected_features
            }
        ).to_csv(
            temporary_directory
            / "final_selected_features.csv",
            index=False,
        )

        prediction_output.to_csv(
            temporary_directory
            / "untouched_test_predictions.csv",
            index=False,
        )

        phase_probabilities.to_csv(
            temporary_directory
            / "phase_level_predictions.csv",
            index=False,
        )

        role_counts.to_csv(
            temporary_directory
            / "participant_phase_role_counts.csv",
            index=False,
        )

        baseline_audit.to_csv(
            temporary_directory
            / "baseline_normalization_audit.csv",
            index=False,
        )

        pd.DataFrame(
            [
                {
                    key: (
                        ";".join(value)
                        if isinstance(
                            value,
                            list,
                        )
                        else value
                    )
                    for key, value
                    in validation_cleanup.items()
                }
            ]
        ).to_csv(
            temporary_directory
            / "validation_cleanup_summary.csv",
            index=False,
        )

        pd.DataFrame(
            [
                {
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
                }
            ]
        ).to_csv(
            temporary_directory
            / "final_cleanup_summary.csv",
            index=False,
        )

        pd.DataFrame(
            final_confusion,
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
            / "untouched_test_confusion_matrix.csv"
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
            / "final_summary.json"
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
        "\n===== UNTOUCHED FINAL TEST RESULTS ====="
    )

    print(
        "Chosen model:",
        chosen_model_name,
    )

    print(
        "Chosen Top-K:",
        int(chosen["top_k"]),
    )

    print(
        "Prototype temperature:",
        float(
            chosen[
                "prototype_temperature"
            ]
        ),
    )

    print(
        "Model probability weight:",
        float(
            chosen[
                "model_probability_weight"
            ]
        ),
    )

    print(
        "Low-class multiplier:",
        float(
            chosen[
                "low_class_multiplier"
            ]
        ),
    )

    print(
        "Probability history:",
        int(
            chosen[
                "probability_history_seconds"
            ]
        ),
        "seconds",
    )

    print(
        "Test decisions:",
        len(final_test_frame),
    )

    print(
        "Accuracy:",
        f"{final_test_metrics['accuracy']:.4f}",
    )

    print(
        "Balanced accuracy:",
        f"{final_test_metrics['balanced_accuracy']:.4f}",
    )

    print(
        "Macro-F1:",
        f"{final_test_metrics['macro_f1']:.4f}",
    )

    print(
        "Macro ROC-AUC:",
        f"{final_test_metrics['macro_roc_auc_ovr']:.4f}",
    )

    print(
        "Macro PR-AUC:",
        f"{final_test_metrics['macro_pr_auc']:.4f}",
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
        final_confusion
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

    print(
        "Output directory:",
        output_directory,
    )

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run validation-selected personalized "
            "three-class final evaluation."
        )
    )

    parser.add_argument(
        "--write",
        action="store_true",
        help="Write all validation and untouched-test artifacts.",
    )

    arguments = parser.parse_args()

    config = load_revision_config(
        check_input_paths=True
    )

    run_personalized_final(
        config,
        write_outputs=arguments.write,
    )


if __name__ == "__main__":
    main()