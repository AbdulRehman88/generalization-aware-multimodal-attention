"""Calibrated temporal three-class attention evaluation.

This revision-only experiment evaluates a personalized deployment protocol:

1. The first 60 seconds of each known low, mid, and high condition are used
   as participant calibration data.
2. A 20-second guard interval separates calibration and test intervals.
3. Features are normalized relative to each participant's initial low-state
   baseline using median and interquartile range.
4. Five consecutive four-second windows form a causal 20-second context.
5. Constant and duplicate removal and SHAP selection use calibration data only.
6. Test decisions are reported at:
   - 20-second contextual prediction level;
   - 20-second causal probability stabilization;
   - 60-second causal probability stabilization;
   - phase/recording level.

This is a subject-calibrated multiclass protocol. It must not be described as
calibration-free or unseen-participant generalization.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from src.core.config import load_revision_config
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


def robust_baseline_normalize(
    frame: pd.DataFrame,
    model_features: list[str],
    calibration_windows_per_phase: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Normalize each participant using initial low-state calibration."""

    transformed = frame.copy()

    # Explicit conversion prevents assignment of normalized floating-point
    # values into integer-typed feature columns.
    transformed[model_features] = transformed[
        model_features
    ].astype(float)

    audit_rows: list[dict[str, Any]] = []

    for participant, indices in (
        transformed.groupby(
            "participant"
        ).groups.items()
    ):
        participant_indices = np.asarray(
            list(indices),
            dtype=int,
        )

        baseline_mask = (
            transformed.loc[
                participant_indices,
                "phase",
            ].to_numpy(dtype=int)
            == 1
        ) & (
            transformed.loc[
                participant_indices,
                "window_index",
            ].to_numpy(dtype=int)
            < calibration_windows_per_phase
        )

        baseline_indices = (
            participant_indices[
                baseline_mask
            ]
        )

        if len(baseline_indices) != calibration_windows_per_phase:
            raise RuntimeError(
                f"{participant}: expected "
                f"{calibration_windows_per_phase} baseline windows, "
                f"found {len(baseline_indices)}."
            )

        baseline_values = transformed.loc[
            baseline_indices,
            model_features,
        ].to_numpy(dtype=float)

        median = np.median(
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
            model_features,
        ].to_numpy(dtype=float)

        normalized = (
            participant_values
            - median
        ) / scale

        transformed.loc[
            participant_indices,
            model_features,
        ] = normalized

        audit_rows.append(
            {
                "participant": participant,
                "baseline_windows":
                    len(baseline_indices),
                "features":
                    len(model_features),
                "zero_iqr_features":
                    int(
                        (
                            interquartile_range
                            <= 1e-9
                        ).sum()
                    ),
                "unit_fallback_features":
                    int(
                        (
                            (interquartile_range <= 1e-9)
                            & (standard_deviation <= 1e-9)
                        ).sum()
                    ),
            }
        )

    values = transformed[
        model_features
    ].to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise RuntimeError(
            "Baseline normalization produced non-finite values."
        )

    return transformed, pd.DataFrame(audit_rows)


def build_causal_context(
    frame: pd.DataFrame,
    model_features: list[str],
    context_windows: int,
) -> pd.DataFrame:
    """Build mean, dispersion, and slope over preceding windows."""

    if context_windows < 2:
        raise ValueError(
            "context_windows must be at least two."
        )

    time_axis = np.arange(
        context_windows,
        dtype=float,
    )

    centered_time = (
        time_axis
        - np.mean(time_axis)
    )

    slope_denominator = float(
        np.sum(
            centered_time ** 2
        )
    )

    context_rows: list[dict[str, Any]] = []

    grouped = frame.groupby(
        [
            "participant",
            "phase",
        ],
        sort=True,
    )

    for (
        participant,
        phase,
    ), group in grouped:
        group = group.sort_values(
            "window_index"
        ).reset_index(drop=True)

        expected_indices = np.arange(
            len(group),
            dtype=int,
        )

        observed_indices = group[
            "window_index"
        ].to_numpy(dtype=int)

        if not np.array_equal(
            observed_indices,
            expected_indices,
        ):
            raise RuntimeError(
                f"Non-contiguous windows for "
                f"{participant}, phase {phase}."
            )

        values = group[
            model_features
        ].to_numpy(dtype=float)

        for current_position in range(
            context_windows - 1,
            len(group),
        ):
            history = values[
                current_position
                - context_windows
                + 1:
                current_position
                + 1
            ]

            history_mean = np.mean(
                history,
                axis=0,
            )

            history_std = np.std(
                history,
                axis=0,
            )

            history_slope = (
                centered_time[:, np.newaxis]
                * history
            ).sum(
                axis=0
            ) / slope_denominator

            source_row = group.iloc[
                current_position
            ]

            row: dict[str, Any] = {
                "segment_id":
                    str(source_row["segment_id"]),
                "participant":
                    str(participant),
                "phase":
                    int(phase),
                "label":
                    int(source_row["label"]),
                "window_index":
                    int(
                        source_row["window_index"]
                    ),
            }

            for feature_index, feature in enumerate(
                model_features
            ):
                row[
                    f"context_mean__{feature}"
                ] = float(
                    history_mean[
                        feature_index
                    ]
                )

                row[
                    f"context_std__{feature}"
                ] = float(
                    history_std[
                        feature_index
                    ]
                )

                row[
                    f"context_slope__{feature}"
                ] = float(
                    history_slope[
                        feature_index
                    ]
                )

            context_rows.append(row)

    context = pd.DataFrame(
        context_rows
    )

    context_features = [
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

    values = context[
        context_features
    ].to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise RuntimeError(
            "Temporal context contains non-finite values."
        )

    return context


def assign_protocol_roles(
    context: pd.DataFrame,
    calibration_windows_per_phase: int,
    guard_windows: int,
) -> pd.Series:
    """Assign calibration, guard, and test roles by time."""

    calibration_end = (
        calibration_windows_per_phase
    )

    test_start = (
        calibration_windows_per_phase
        + guard_windows
    )

    window_indices = context[
        "window_index"
    ].to_numpy(dtype=int)

    roles = np.where(
        window_indices < calibration_end,
        "calibration",
        np.where(
            window_indices < test_start,
            "guard",
            "test",
        ),
    )

    return pd.Series(
        roles,
        index=context.index,
        dtype="object",
    )


def build_models(
    config: dict[str, Any],
    random_seed: int,
) -> dict[str, Any]:
    """Build the two fixed pilot classifiers."""

    extra_parameters = config[
        "models"
    ][
        "extra_trees"
    ]

    xgb_parameters = config[
        "models"
    ][
        "xgboost"
    ]

    return {
        "extra_trees": ExtraTreesClassifier(
            n_estimators=int(
                extra_parameters[
                    "n_estimators"
                ]
            ),
            max_features=str(
                extra_parameters[
                    "max_features"
                ]
            ),
            min_samples_leaf=int(
                extra_parameters[
                    "min_samples_leaf"
                ]
            ),
            class_weight=str(
                extra_parameters[
                    "class_weight"
                ]
            ),
            n_jobs=-1,
            random_state=random_seed,
        ),
        "xgboost": XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            tree_method="hist",
            n_jobs=-1,
            random_state=random_seed,
            n_estimators=int(
                xgb_parameters[
                    "n_estimators"
                ]
            ),
            max_depth=int(
                xgb_parameters[
                    "max_depth"
                ]
            ),
            learning_rate=float(
                xgb_parameters[
                    "learning_rate"
                ]
            ),
            subsample=float(
                xgb_parameters[
                    "subsample"
                ]
            ),
            colsample_bytree=float(
                xgb_parameters[
                    "colsample_bytree"
                ]
            ),
            min_child_weight=float(
                xgb_parameters[
                    "min_child_weight"
                ]
            ),
            reg_lambda=float(
                xgb_parameters[
                    "reg_lambda"
                ]
            ),
        ),
    }


def smooth_probabilities(
    predictions: pd.DataFrame,
    smoothing_windows: int,
) -> pd.DataFrame:
    """Apply causal probability averaging within participant and phase."""

    result = predictions.copy()

    probability_columns = [
        "probability_class_0",
        "probability_class_1",
        "probability_class_2",
    ]

    result[
        [
            "smoothed_probability_class_0",
            "smoothed_probability_class_1",
            "smoothed_probability_class_2",
        ]
    ] = np.nan

    grouped = result.groupby(
        [
            "participant",
            "phase",
        ],
        sort=False,
    )

    for _, indices in grouped.groups.items():
        indices = list(indices)

        ordered_indices = (
            result.loc[
                indices
            ]
            .sort_values(
                "window_index"
            )
            .index
        )

        probabilities = result.loc[
            ordered_indices,
            probability_columns,
        ]

        smoothed = probabilities.rolling(
            window=smoothing_windows,
            min_periods=1,
        ).mean()

        result.loc[
            ordered_indices,
            [
                "smoothed_probability_class_0",
                "smoothed_probability_class_1",
                "smoothed_probability_class_2",
            ],
        ] = smoothed.to_numpy(dtype=float)

    smoothed_columns = [
        "smoothed_probability_class_0",
        "smoothed_probability_class_1",
        "smoothed_probability_class_2",
    ]

    result[
        "smoothed_prediction"
    ] = np.argmax(
        result[
            smoothed_columns
        ].to_numpy(dtype=float),
        axis=1,
    )

    return result


def evaluate_rows(
    frame: pd.DataFrame,
    prediction_column: str,
    probability_columns: list[str],
) -> dict[str, float]:
    """Evaluate predictions on a prepared row set."""

    return compute_metrics(
        frame[
            "true_label"
        ].to_numpy(dtype=int),
        frame[
            prediction_column
        ].to_numpy(dtype=int),
        frame[
            probability_columns
        ].to_numpy(dtype=float),
    )


def run_calibrated_temporal_pilot(
    config: dict[str, Any],
    *,
    write_outputs: bool,
) -> dict[str, Any]:
    """Run calibrated temporal three-class classification."""

    training = config[
        "training"
    ]

    pilot = training[
        "calibrated_temporal_pilot"
    ]

    modality = str(
        pilot["modality"]
    )

    calibration_windows = int(
        pilot[
            "calibration_windows_per_phase"
        ]
    )

    guard_windows = int(
        pilot["guard_windows"]
    )

    context_windows = int(
        pilot["context_windows"]
    )

    top_k = int(
        pilot["top_k"]
    )

    shap_max_samples = int(
        pilot["shap_max_samples"]
    )

    smoothing_windows = [
        int(value)
        for value in pilot[
            "probability_smoothing_windows"
        ]
    ]

    random_seed = int(
        training["random_seed"]
    )

    feature_directory = resolve_project_path(
        config["features"]["output_dir"]
    )

    feature_path = (
        feature_directory
        / f"{modality}_features.csv"
    )

    if not feature_path.is_file():
        raise FileNotFoundError(
            f"Feature matrix not found: {feature_path}"
        )

    output_directory = resolve_project_path(
        pilot["output_dir"]
    )

    frame = pd.read_csv(
        feature_path,
        low_memory=False,
    )

    model_features = feature_columns(
        frame
    )

    normalized, baseline_audit = (
        robust_baseline_normalize(
            frame,
            model_features,
            calibration_windows,
        )
    )

    print(
        "Building causal temporal context...",
        flush=True,
    )

    context = build_causal_context(
        normalized,
        model_features,
        context_windows,
    )

    context["role"] = assign_protocol_roles(
        context,
        calibration_windows,
        guard_windows,
    )

    training_frame = context.loc[
        context["role"]
        == "calibration"
    ].copy()

    inference_frame = context.loc[
        context["role"]
        .isin(
            [
                "guard",
                "test",
            ]
        )
    ].copy()

    test_frame = context.loc[
        context["role"]
        == "test"
    ].copy()

    if training_frame.empty or test_frame.empty:
        raise RuntimeError(
            "Calibration or test partition is empty."
        )

    if set(
        training_frame[
            "segment_id"
        ]
    ).intersection(
        set(
            test_frame[
                "segment_id"
            ]
        )
    ):
        raise RuntimeError(
            "Calibration/test segment overlap detected."
        )

    participant_phase_role_counts = (
        context.groupby(
            [
                "participant",
                "phase",
                "role",
            ]
        )
        .size()
        .rename("windows")
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

    retained_features, cleanup = (
        fit_training_cleanup(
            training_frame[
                temporal_features
            ]
        )
    )

    selector_parameters = training[
        "shap_selector"
    ]

    print(
        "Computing calibration-only SHAP ranking...",
        flush=True,
    )

    ranking = select_training_fold_features(
        training_frame[
            retained_features
        ],
        training_frame[
            "label"
        ].to_numpy(dtype=int),
        top_k=top_k,
        random_seed=random_seed,
        maximum_shap_samples=shap_max_samples,
        selector_parameters=selector_parameters,
    )

    selected_features = (
        ranking.loc[
            ranking["selected"],
            "feature",
        ]
        .tolist()
    )

    if not selected_features:
        raise RuntimeError(
            "No temporal features were selected."
        )

    models = build_models(
        config,
        random_seed,
    )

    metric_rows: list[dict[str, Any]] = []
    prediction_tables: list[pd.DataFrame] = []
    confusion_matrices: dict[str, Any] = {}

    probability_columns = [
        "probability_class_0",
        "probability_class_1",
        "probability_class_2",
    ]

    smoothed_probability_columns = [
        "smoothed_probability_class_0",
        "smoothed_probability_class_1",
        "smoothed_probability_class_2",
    ]

    for model_name, model in models.items():
        x_train = training_frame[
            selected_features
        ]

        y_train = training_frame[
            "label"
        ].to_numpy(dtype=int)

        x_inference = inference_frame[
            selected_features
        ]

        sample_weights = compute_sample_weight(
            class_weight="balanced",
            y=y_train,
        )

        if model_name == "xgboost":
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

        probabilities = model.predict_proba(
            x_inference
        )

        if not np.array_equal(
            model.classes_,
            CLASS_LABELS,
        ):
            raise RuntimeError(
                f"{model_name}: unexpected classes "
                f"{model.classes_}."
            )

        predictions = np.argmax(
            probabilities,
            axis=1,
        )

        model_predictions = inference_frame[
            [
                "segment_id",
                "participant",
                "phase",
                "label",
                "window_index",
                "role",
            ]
        ].copy()

        model_predictions = model_predictions.rename(
            columns={
                "label": "true_label",
            }
        )

        model_predictions[
            "model"
        ] = model_name

        model_predictions[
            "prediction"
        ] = predictions

        for class_index, column in enumerate(
            probability_columns
        ):
            model_predictions[
                column
            ] = probabilities[
                :,
                class_index,
            ]

        for smoothing in smoothing_windows:
            stabilized = smooth_probabilities(
                model_predictions,
                smoothing,
            )

            evaluated = stabilized.loc[
                stabilized["role"]
                == "test"
            ]

            metrics = evaluate_rows(
                evaluated,
                "smoothed_prediction",
                smoothed_probability_columns,
            )

            endpoint = (
                "contextual_20_second"
                if smoothing == 1
                else (
                    f"causal_probability_"
                    f"{smoothing * 4}_second"
                )
            )

            metric_rows.append(
                {
                    "model": model_name,
                    "endpoint": endpoint,
                    "context_seconds":
                        context_windows * 4,
                    "probability_history_seconds":
                        smoothing * 4,
                    "decision_count":
                        len(evaluated),
                    **metrics,
                }
            )

            confusion = pd.crosstab(
                evaluated[
                    "true_label"
                ],
                evaluated[
                    "smoothed_prediction"
                ],
            ).reindex(
                index=[0, 1, 2],
                columns=[0, 1, 2],
                fill_value=0,
            )

            confusion_matrices[
                f"{model_name}__{endpoint}"
            ] = confusion.to_numpy(
                dtype=int
            ).tolist()

            saved = evaluated.copy()

            saved[
                "endpoint"
            ] = endpoint

            prediction_tables.append(
                saved
            )

        test_predictions = model_predictions.loc[
            model_predictions["role"]
            == "test"
        ].copy()

        phase_probabilities = (
            test_predictions.groupby(
                [
                    "participant",
                    "phase",
                    "true_label",
                ],
                as_index=False,
            )[
                probability_columns
            ]
            .mean()
        )

        phase_probabilities[
            "phase_prediction"
        ] = np.argmax(
            phase_probabilities[
                probability_columns
            ].to_numpy(dtype=float),
            axis=1,
        )

        phase_metrics = evaluate_rows(
            phase_probabilities,
            "phase_prediction",
            probability_columns,
        )

        metric_rows.append(
            {
                "model": model_name,
                "endpoint":
                    "phase_recording_mean_probability",
                "context_seconds":
                    context_windows * 4,
                "probability_history_seconds":
                    "full_test_phase",
                "decision_count":
                    len(phase_probabilities),
                **phase_metrics,
            }
        )

        phase_confusion = pd.crosstab(
            phase_probabilities[
                "true_label"
            ],
            phase_probabilities[
                "phase_prediction"
            ],
        ).reindex(
            index=[0, 1, 2],
            columns=[0, 1, 2],
            fill_value=0,
        )

        confusion_matrices[
            (
                f"{model_name}__"
                "phase_recording_mean_probability"
            )
        ] = phase_confusion.to_numpy(
            dtype=int
        ).tolist()

        phase_probabilities[
            "model"
        ] = model_name

        phase_probabilities[
            "endpoint"
        ] = (
            "phase_recording_mean_probability"
        )

        prediction_tables.append(
            phase_probabilities
        )

        print(
            f"Completed model: {model_name}",
            flush=True,
        )

    metrics = pd.DataFrame(
        metric_rows
    ).sort_values(
        [
            "accuracy",
            "macro_f1",
        ],
        ascending=False,
        ignore_index=True,
    )

    best_accuracy = metrics.iloc[0].to_dict()

    best_macro_f1 = (
        metrics.sort_values(
            "macro_f1",
            ascending=False,
        )
        .iloc[0]
        .to_dict()
    )

    selected_stability = ranking.copy()

    summary = {
        "experiment_type":
            "subject_calibrated_temporal_multiclass_pilot",
        "classification_task":
            "three_class_low_mid_high",
        "protocol_scope":
            "within_cohort_subject_calibrated",
        "calibration_free":
            False,
        "unseen_participant_generalization":
            False,
        "modality": modality,
        "participants":
            sorted(
                frame[
                    "participant"
                ].astype(str).unique().tolist()
            ),
        "participant_count":
            int(
                frame[
                    "participant"
                ].nunique()
            ),
        "calibration_windows_per_phase":
            calibration_windows,
        "calibration_seconds_per_phase":
            calibration_windows * 4,
        "guard_windows":
            guard_windows,
        "guard_seconds":
            guard_windows * 4,
        "context_windows":
            context_windows,
        "context_seconds":
            context_windows * 4,
        "input_features":
            len(model_features),
        "temporal_input_features":
            len(temporal_features),
        "retained_after_training_cleanup":
            len(retained_features),
        "selected_features":
            len(selected_features),
        "calibration_rows":
            int(len(training_frame)),
        "guard_rows":
            int(
                (
                    context["role"]
                    == "guard"
                ).sum()
            ),
        "test_rows":
            int(len(test_frame)),
        "best_accuracy_result":
            best_accuracy,
        "best_macro_f1_result":
            best_macro_f1,
        "all_results":
            metrics.to_dict(
                orient="records"
            ),
        "confusion_matrices":
            confusion_matrices,
        "interpretation_constraint":
            (
                "Results represent a subject-calibrated "
                "personalized monitoring protocol and must not "
                "be described as calibration-free "
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
                "Refusing to overwrite existing calibrated "
                f"pilot output: {output_directory}"
            )

        if temporary_directory.exists():
            raise FileExistsError(
                f"Temporary output exists: {temporary_directory}"
            )

        temporary_directory.mkdir(
            parents=True,
            exist_ok=False,
        )

        metrics.to_csv(
            temporary_directory
            / "endpoint_metrics.csv",
            index=False,
        )

        ranking.to_csv(
            temporary_directory
            / "calibration_shap_ranking.csv",
            index=False,
        )

        baseline_audit.to_csv(
            temporary_directory
            / "baseline_normalization_audit.csv",
            index=False,
        )

        participant_phase_role_counts.to_csv(
            temporary_directory
            / "participant_phase_role_counts.csv",
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
                    for key, value in cleanup.items()
                }
            ]
        ).to_csv(
            temporary_directory
            / "training_cleanup_summary.csv",
            index=False,
        )

        selected_stability.loc[
            selected_stability["selected"]
        ].to_csv(
            temporary_directory
            / "selected_temporal_features.csv",
            index=False,
        )

        for index, table in enumerate(
            prediction_tables,
            start=1,
        ):
            table.to_csv(
                temporary_directory
                / f"predictions_{index:02d}.csv",
                index=False,
            )

        (
            temporary_directory
            / "calibrated_temporal_summary.json"
        ).write_text(
            json.dumps(
                summary,
                indent=2,
                sort_keys=True,
                default=str,
            )
            + "\n",
            encoding="utf-8",
        )

        temporary_directory.replace(
            output_directory
        )

    print(
        "\n===== CALIBRATED TEMPORAL RESULTS ====="
    )

    display_columns = [
        "model",
        "endpoint",
        "decision_count",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "macro_roc_auc_ovr",
        "macro_pr_auc",
    ]

    print(
        metrics[
            display_columns
        ].to_string(
            index=False
        )
    )

    print(
        "\nBest accuracy:",
        f"{float(best_accuracy['accuracy']):.4f}",
        "|",
        best_accuracy["model"],
        "|",
        best_accuracy["endpoint"],
    )

    print(
        "Best Macro-F1:",
        f"{float(best_macro_f1['macro_f1']):.4f}",
        "|",
        best_macro_f1["model"],
        "|",
        best_macro_f1["endpoint"],
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
            "Run calibrated temporal multiclass pilot."
        )
    )

    parser.add_argument(
        "--write",
        action="store_true",
        help="Write all evaluation artifacts.",
    )

    arguments = parser.parse_args()

    config = load_revision_config(
        check_input_paths=True
    )

    run_calibrated_temporal_pilot(
        config,
        write_outputs=arguments.write,
    )


if __name__ == "__main__":
    main()
