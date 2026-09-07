"""Leakage-safe participant-grouped pilot evaluation.

This revision-only pilot validates the complete training and testing workflow:

- Five-fold participant-disjoint StratifiedGroupKFold evaluation.
- No participant overlap between training and test data.
- Constant and duplicate feature removal fitted on training rows only.
- SHAP feature ranking fitted on training rows only.
- Extra Trees classification with balanced class weighting.
- Test predictions generated exactly once for every eligible window.

This pilot uses fixed parameters and must not be interpreted as the final
model-selection or hyperparameter-tuning experiment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import label_binarize
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from src.core.config import load_revision_config


PROJECT_ROOT = Path(__file__).resolve().parents[2]

METADATA_COLUMNS = {
    "segment_id",
    "participant",
    "phase",
    "label",
    "window_index",
    "pupil_available",
}

CLASS_LABELS = np.asarray([0, 1, 2], dtype=int)


def resolve_project_path(value: str) -> Path:
    """Resolve a project-relative or absolute path."""

    path = Path(value)

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path.resolve()


def feature_columns(frame: pd.DataFrame) -> list[str]:
    """Return numerical model-feature columns."""

    return [
        column
        for column in frame.columns
        if column not in METADATA_COLUMNS
    ]


def fit_training_cleanup(
    training_frame: pd.DataFrame,
) -> tuple[list[str], dict[str, Any]]:
    """Fit constant and exact-duplicate removal using training rows only."""

    if training_frame.empty:
        raise ValueError("Training feature frame is empty.")

    constant_columns = [
        column
        for column in training_frame.columns
        if training_frame[column].nunique(dropna=False) <= 1
    ]

    nonconstant_columns = [
        column
        for column in training_frame.columns
        if column not in constant_columns
    ]

    if not nonconstant_columns:
        raise RuntimeError(
            "All training features were constant."
        )

    duplicate_mask = (
        training_frame[
            nonconstant_columns
        ]
        .T
        .duplicated(keep="first")
    )

    duplicate_columns = (
        duplicate_mask[
            duplicate_mask
        ]
        .index
        .tolist()
    )

    retained_columns = [
        column
        for column in nonconstant_columns
        if column not in duplicate_columns
    ]

    if not retained_columns:
        raise RuntimeError(
            "No features remained after training-fold cleanup."
        )

    summary = {
        "input_features": len(training_frame.columns),
        "constant_removed": len(constant_columns),
        "duplicate_removed": len(duplicate_columns),
        "retained_features": len(retained_columns),
        "constant_columns": constant_columns,
        "duplicate_columns": duplicate_columns,
    }

    return retained_columns, summary


def normalize_shap_importance(
    values: Any,
    feature_count: int,
) -> np.ndarray:
    """Convert common SHAP multiclass outputs into one importance vector."""

    if isinstance(values, list):
        arrays = [
            np.asarray(value, dtype=float)
            for value in values
        ]

        if not arrays:
            raise ValueError("SHAP returned an empty list.")

        stacked = np.stack(
            arrays,
            axis=-1,
        )

        if stacked.ndim != 3:
            raise ValueError(
                f"Unexpected stacked SHAP dimensions: {stacked.shape}"
            )

        importance = np.mean(
            np.abs(stacked),
            axis=(0, 2),
        )

    else:
        array = np.asarray(
            values,
            dtype=float,
        )

        if array.ndim == 2:
            if array.shape[1] != feature_count:
                raise ValueError(
                    f"Unexpected SHAP shape: {array.shape}"
                )

            importance = np.mean(
                np.abs(array),
                axis=0,
            )

        elif array.ndim == 3:
            if array.shape[1] == feature_count:
                importance = np.mean(
                    np.abs(array),
                    axis=(0, 2),
                )

            elif array.shape[2] == feature_count:
                importance = np.mean(
                    np.abs(array),
                    axis=(0, 1),
                )

            else:
                raise ValueError(
                    f"Cannot identify feature axis in SHAP shape "
                    f"{array.shape}."
                )

        else:
            raise ValueError(
                f"Unexpected SHAP output dimensions: {array.shape}"
            )

    importance = np.asarray(
        importance,
        dtype=float,
    ).reshape(-1)

    if len(importance) != feature_count:
        raise ValueError(
            f"Expected {feature_count} SHAP importances, "
            f"received {len(importance)}."
        )

    if not np.isfinite(importance).all():
        raise ValueError(
            "SHAP importance contains non-finite values."
        )

    return importance


def select_training_fold_features(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    *,
    top_k: int,
    random_seed: int,
    maximum_shap_samples: int,
    selector_parameters: dict[str, Any],
) -> pd.DataFrame:
    """Rank features using only the outer training fold."""

    if top_k <= 0:
        raise ValueError("top_k must be positive.")

    selector = XGBClassifier(
        objective="multi:softprob",
        num_class=len(CLASS_LABELS),
        eval_metric="mlogloss",
        tree_method="hist",
        n_jobs=-1,
        random_state=random_seed,
        n_estimators=int(
            selector_parameters["n_estimators"]
        ),
        max_depth=int(
            selector_parameters["max_depth"]
        ),
        learning_rate=float(
            selector_parameters["learning_rate"]
        ),
        subsample=float(
            selector_parameters["subsample"]
        ),
        colsample_bytree=float(
            selector_parameters["colsample_bytree"]
        ),
    )

    sample_weights = compute_sample_weight(
        class_weight="balanced",
        y=y_train,
    )

    selector.fit(
        x_train,
        y_train,
        sample_weight=sample_weights,
    )

    if len(x_train) > maximum_shap_samples:
        rng = np.random.default_rng(
            random_seed
        )

        sampled_indices = []

        for class_label in CLASS_LABELS:
            class_indices = np.flatnonzero(
                y_train == class_label
            )

            class_target = max(
                1,
                int(
                    round(
                        maximum_shap_samples
                        * len(class_indices)
                        / len(y_train)
                    )
                ),
            )

            class_target = min(
                class_target,
                len(class_indices),
            )

            sampled_indices.extend(
                rng.choice(
                    class_indices,
                    size=class_target,
                    replace=False,
                ).tolist()
            )

        sampled_indices = np.asarray(
            sampled_indices,
            dtype=int,
        )

        if len(sampled_indices) > maximum_shap_samples:
            sampled_indices = rng.choice(
                sampled_indices,
                size=maximum_shap_samples,
                replace=False,
            )

        shap_frame = x_train.iloc[
            np.sort(sampled_indices)
        ]

    else:
        shap_frame = x_train

    explainer = shap.TreeExplainer(
        selector
    )

    shap_values = explainer.shap_values(
        shap_frame
    )

    importance = normalize_shap_importance(
        shap_values,
        feature_count=x_train.shape[1],
    )

    ranking = pd.DataFrame(
        {
            "feature": x_train.columns,
            "mean_absolute_shap": importance,
        }
    ).sort_values(
        [
            "mean_absolute_shap",
            "feature",
        ],
        ascending=[False, True],
        ignore_index=True,
    )

    ranking["rank"] = (
        np.arange(len(ranking))
        + 1
    )

    ranking["selected"] = (
        ranking["rank"]
        <= min(top_k, len(ranking))
    )

    return ranking


def build_grouped_splits(
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    folds: int,
    random_seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Build deterministic participant-disjoint outer folds."""

    splitter = StratifiedGroupKFold(
        n_splits=folds,
        shuffle=True,
        random_state=random_seed,
    )

    dummy = np.zeros(
        (len(labels), 1),
        dtype=float,
    )

    splits = list(
        splitter.split(
            dummy,
            labels,
            groups,
        )
    )

    test_counts = np.zeros(
        len(labels),
        dtype=np.int16,
    )

    for train_indices, test_indices in splits:
        train_groups = set(
            groups[train_indices]
        )

        test_groups = set(
            groups[test_indices]
        )

        overlap = (
            train_groups
            & test_groups
        )

        if overlap:
            raise RuntimeError(
                f"Participant leakage detected: {sorted(overlap)}"
            )

        if set(np.unique(labels[train_indices])) != set(CLASS_LABELS):
            raise RuntimeError(
                "An outer training fold lacks one or more classes."
            )

        if set(np.unique(labels[test_indices])) != set(CLASS_LABELS):
            raise RuntimeError(
                "An outer test fold lacks one or more classes."
            )

        test_counts[test_indices] += 1

    if not np.all(test_counts == 1):
        raise RuntimeError(
            "Every row must appear in exactly one outer test fold."
        )

    return splits


def compute_metrics(
    y_true: np.ndarray,
    y_prediction: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float]:
    """Compute primary and companion multiclass metrics."""

    binary_truth = label_binarize(
        y_true,
        classes=CLASS_LABELS,
    )

    return {
        "accuracy": float(
            accuracy_score(
                y_true,
                y_prediction,
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_true,
                y_prediction,
            )
        ),
        "macro_f1": float(
            f1_score(
                y_true,
                y_prediction,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_precision": float(
            precision_score(
                y_true,
                y_prediction,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_recall": float(
            recall_score(
                y_true,
                y_prediction,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_roc_auc_ovr": float(
            roc_auc_score(
                binary_truth,
                probabilities,
                average="macro",
                multi_class="ovr",
            )
        ),
        "macro_pr_auc": float(
            average_precision_score(
                binary_truth,
                probabilities,
                average="macro",
            )
        ),
    }


def run_internal_grouped_pilot(
    config: dict[str, Any],
    *,
    write_outputs: bool,
) -> dict[str, Any]:
    """Run the leakage-safe ECG-EEG pilot experiment."""

    training = config["training"]
    pilot = training["internal_pilot"]

    modality = str(
        pilot["modality"]
    )

    top_k = int(
        pilot["top_k"]
    )

    shap_max_samples = int(
        pilot["shap_max_samples"]
    )

    random_seed = int(
        training["random_seed"]
    )

    folds = int(
        config[
            "evaluation"
        ][
            "internal"
        ][
            "primary"
        ][
            "folds"
        ]
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

    required_metadata = {
        "segment_id",
        "participant",
        "label",
    }

    missing_metadata = (
        required_metadata
        - set(frame.columns)
    )

    if missing_metadata:
        raise RuntimeError(
            f"Feature matrix lacks metadata: "
            f"{sorted(missing_metadata)}"
        )

    all_features = feature_columns(
        frame
    )

    if not all_features:
        raise RuntimeError(
            "Feature matrix contains no model features."
        )

    feature_values = frame[
        all_features
    ].to_numpy(dtype=float)

    if not np.isfinite(feature_values).all():
        raise RuntimeError(
            "Input feature matrix contains non-finite values."
        )

    labels = frame[
        "label"
    ].to_numpy(dtype=int)

    groups = frame[
        "participant"
    ].astype(str).to_numpy()

    splits = build_grouped_splits(
        labels,
        groups,
        folds=folds,
        random_seed=random_seed,
    )

    selector_parameters = training[
        "shap_selector"
    ]

    model_parameters = config[
        "models"
    ][
        "extra_trees"
    ]

    fold_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    ranking_rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    cleanup_rows: list[dict[str, Any]] = []

    for fold_number, (
        train_indices,
        test_indices,
    ) in enumerate(
        splits,
        start=1,
    ):
        training_participants = sorted(
            set(groups[train_indices])
        )

        test_participants = sorted(
            set(groups[test_indices])
        )

        x_train_full = frame.iloc[
            train_indices
        ][
            all_features
        ]

        x_test_full = frame.iloc[
            test_indices
        ][
            all_features
        ]

        y_train = labels[
            train_indices
        ]

        y_test = labels[
            test_indices
        ]

        retained_columns, cleanup = (
            fit_training_cleanup(
                x_train_full
            )
        )

        x_train_clean = x_train_full[
            retained_columns
        ]

        x_test_clean = x_test_full[
            retained_columns
        ]

        ranking = select_training_fold_features(
            x_train_clean,
            y_train,
            top_k=top_k,
            random_seed=(
                random_seed
                + fold_number
            ),
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
                f"Fold {fold_number} selected no features."
            )

        model = ExtraTreesClassifier(
            n_estimators=int(
                model_parameters["n_estimators"]
            ),
            max_features=str(
                model_parameters["max_features"]
            ),
            min_samples_leaf=int(
                model_parameters["min_samples_leaf"]
            ),
            class_weight=str(
                model_parameters["class_weight"]
            ),
            n_jobs=-1,
            random_state=(
                random_seed
                + 1000
                + fold_number
            ),
        )

        model.fit(
            x_train_clean[
                selected_features
            ],
            y_train,
        )

        predictions = model.predict(
            x_test_clean[
                selected_features
            ]
        )

        probabilities = model.predict_proba(
            x_test_clean[
                selected_features
            ]
        )

        if not np.array_equal(
            model.classes_,
            CLASS_LABELS,
        ):
            raise RuntimeError(
                f"Unexpected model classes: {model.classes_}"
            )

        metrics = compute_metrics(
            y_test,
            predictions,
            probabilities,
        )

        fold_rows.append(
            {
                "fold": fold_number,
                "modality": modality,
                "model": "extra_trees",
                "top_k": top_k,
                "training_participants":
                    ";".join(
                        training_participants
                    ),
                "test_participants":
                    ";".join(
                        test_participants
                    ),
                "training_participant_count":
                    len(training_participants),
                "test_participant_count":
                    len(test_participants),
                "training_rows":
                    len(train_indices),
                "test_rows":
                    len(test_indices),
                "input_features":
                    len(all_features),
                "retained_after_cleanup":
                    len(retained_columns),
                "selected_features":
                    len(selected_features),
                **metrics,
            }
        )

        cleanup_rows.append(
            {
                "fold": fold_number,
                **{
                    key: value
                    for key, value
                    in cleanup.items()
                    if key not in {
                        "constant_columns",
                        "duplicate_columns",
                    }
                },
                "constant_columns":
                    ";".join(
                        cleanup[
                            "constant_columns"
                        ]
                    ),
                "duplicate_columns":
                    ";".join(
                        cleanup[
                            "duplicate_columns"
                        ]
                    ),
            }
        )

        for ranking_row in ranking.itertuples(
            index=False
        ):
            ranking_rows.append(
                {
                    "fold": fold_number,
                    "feature":
                        ranking_row.feature,
                    "mean_absolute_shap":
                        float(
                            ranking_row.mean_absolute_shap
                        ),
                    "rank":
                        int(ranking_row.rank),
                    "selected":
                        bool(ranking_row.selected),
                }
            )

        test_frame = frame.iloc[
            test_indices
        ]

        for local_index, (
            row,
            predicted,
            probability,
        ) in enumerate(
            zip(
                test_frame.itertuples(
                    index=False
                ),
                predictions,
                probabilities,
            )
        ):
            prediction_rows.append(
                {
                    "fold": fold_number,
                    "segment_id":
                        str(row.segment_id),
                    "participant":
                        str(row.participant),
                    "phase":
                        int(row.phase),
                    "true_label":
                        int(y_test[local_index]),
                    "predicted_label":
                        int(predicted),
                    "probability_class_0":
                        float(probability[0]),
                    "probability_class_1":
                        float(probability[1]),
                    "probability_class_2":
                        float(probability[2]),
                }
            )

            split_rows.append(
                {
                    "fold": fold_number,
                    "segment_id":
                        str(row.segment_id),
                    "participant":
                        str(row.participant),
                    "label":
                        int(y_test[local_index]),
                    "role": "test",
                }
            )

        print(
            f"Fold {fold_number}/{folds}: "
            f"test participants={test_participants}, "
            f"Macro-F1={metrics['macro_f1']:.4f}, "
            f"BA={metrics['balanced_accuracy']:.4f}",
            flush=True,
        )

    fold_results = pd.DataFrame(
        fold_rows
    )

    predictions = pd.DataFrame(
        prediction_rows
    )

    rankings = pd.DataFrame(
        ranking_rows
    )

    split_manifest = pd.DataFrame(
        split_rows
    )

    cleanup_results = pd.DataFrame(
        cleanup_rows
    )

    if predictions["segment_id"].duplicated().any():
        raise RuntimeError(
            "A test segment was predicted more than once."
        )

    if len(predictions) != len(frame):
        raise RuntimeError(
            f"Expected {len(frame)} pooled predictions, "
            f"received {len(predictions)}."
        )

    if (
        set(predictions["participant"])
        != set(frame["participant"].astype(str))
    ):
        raise RuntimeError(
            "Not all participants are represented in pooled tests."
        )

    pooled_probabilities = predictions[
        [
            "probability_class_0",
            "probability_class_1",
            "probability_class_2",
        ]
    ].to_numpy(dtype=float)

    pooled_metrics = compute_metrics(
        predictions[
            "true_label"
        ].to_numpy(dtype=int),
        predictions[
            "predicted_label"
        ].to_numpy(dtype=int),
        pooled_probabilities,
    )

    pooled_confusion = confusion_matrix(
        predictions["true_label"],
        predictions["predicted_label"],
        labels=CLASS_LABELS,
    )

    metric_columns = [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "macro_precision",
        "macro_recall",
        "macro_roc_auc_ovr",
        "macro_pr_auc",
    ]

    fold_metric_summary = {
        metric: {
            "mean": float(
                fold_results[metric].mean()
            ),
            "standard_deviation": float(
                fold_results[metric].std(
                    ddof=1
                )
            ),
            "minimum": float(
                fold_results[metric].min()
            ),
            "maximum": float(
                fold_results[metric].max()
            ),
        }
        for metric in metric_columns
    }

    selected_stability = (
        rankings.loc[
            rankings["selected"]
        ]
        .groupby("feature")
        .agg(
            selected_fold_count=(
                "fold",
                "nunique",
            ),
            mean_rank=(
                "rank",
                "mean",
            ),
            mean_absolute_shap=(
                "mean_absolute_shap",
                "mean",
            ),
        )
        .reset_index()
        .sort_values(
            [
                "selected_fold_count",
                "mean_rank",
                "feature",
            ],
            ascending=[
                False,
                True,
                True,
            ],
            ignore_index=True,
        )
    )

    summary = {
        "experiment_type":
            "participant_grouped_internal_pilot",
        "status":
            "pipeline_validation_not_final_model_selection",
        "modality": modality,
        "model": "extra_trees",
        "outer_protocol":
            "StratifiedGroupKFold",
        "group_unit": "participant",
        "folds": folds,
        "random_seed": random_seed,
        "top_k": top_k,
        "shap_selection_scope":
            "outer_training_fold_only",
        "cleanup_scope":
            "outer_training_fold_only",
        "standardization_applied": False,
        "standardization_rationale":
            "Tree-based selector and classifier do not require scaling.",
        "participants": sorted(
            frame[
                "participant"
            ].astype(str).unique().tolist()
        ),
        "participant_count": int(
            frame[
                "participant"
            ].nunique()
        ),
        "rows": int(len(frame)),
        "input_feature_count":
            len(all_features),
        "fold_metrics":
            fold_metric_summary,
        "pooled_metrics":
            pooled_metrics,
        "pooled_confusion_matrix":
            pooled_confusion.tolist(),
        "test_prediction_count":
            int(len(predictions)),
        "unique_test_segment_count":
            int(
                predictions[
                    "segment_id"
                ].nunique()
            ),
        "participant_overlap_detected":
            False,
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
                "Refusing to overwrite existing pilot output: "
                f"{output_directory}"
            )

        if temporary_directory.exists():
            raise FileExistsError(
                f"Temporary output exists: {temporary_directory}"
            )

        temporary_directory.mkdir(
            parents=True,
            exist_ok=False,
        )

        fold_results.to_csv(
            temporary_directory
            / "fold_metrics.csv",
            index=False,
        )

        predictions.to_csv(
            temporary_directory
            / "test_predictions.csv",
            index=False,
        )

        rankings.to_csv(
            temporary_directory
            / "fold_shap_rankings.csv",
            index=False,
        )

        selected_stability.to_csv(
            temporary_directory
            / "selected_feature_stability.csv",
            index=False,
        )

        split_manifest.to_csv(
            temporary_directory
            / "test_fold_manifest.csv",
            index=False,
        )

        cleanup_results.to_csv(
            temporary_directory
            / "fold_cleanup_summary.csv",
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

        (
            temporary_directory
            / "pilot_summary.json"
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

    print("\n===== PARTICIPANT-GROUPED PILOT SUMMARY =====")
    print("Rows:", summary["rows"])
    print("Participants:", summary["participant_count"])
    print("Input features:", summary["input_feature_count"])
    print(
        "Pooled Macro-F1:",
        f"{pooled_metrics['macro_f1']:.4f}",
    )
    print(
        "Pooled balanced accuracy:",
        f"{pooled_metrics['balanced_accuracy']:.4f}",
    )
    print(
        "Pooled accuracy:",
        f"{pooled_metrics['accuracy']:.4f}",
    )
    print(
        "Pooled ROC-AUC:",
        f"{pooled_metrics['macro_roc_auc_ovr']:.4f}",
    )
    print(
        "Participant overlap detected:",
        summary["participant_overlap_detected"],
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
            "Run leakage-safe participant-grouped "
            "internal XR pilot training."
        )
    )

    parser.add_argument(
        "--write",
        action="store_true",
        help="Write fold metrics and predictions.",
    )

    arguments = parser.parse_args()

    config = load_revision_config(
        check_input_paths=True
    )

    run_internal_grouped_pilot(
        config,
        write_outputs=arguments.write,
    )


if __name__ == "__main__":
    main()