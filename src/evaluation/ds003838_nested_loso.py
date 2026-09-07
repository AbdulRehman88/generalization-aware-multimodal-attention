"""Leakage-safe nested LOSO evaluation for ds003838 ECG+EEG."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from xgboost import XGBClassifier

from src.core.config import load_revision_config
from src.evaluation.internal_grouped_pilot import (
    CLASS_LABELS,
    compute_metrics,
)
from src.evaluation.internal_nested_loso_long_windows import (
    deterministic_inner_split,
    fit_classifier,
    fit_training_ranking,
    selected_features_for_mode,
)
from src.features.internal_xr_revision_features import (
    feature_columns,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_project_path(
    value: str | Path,
) -> Path:
    path = Path(value)

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path.resolve()


def validate_disjoint_roles(
    *,
    outer_test_participant: str,
    inner_training_participants: list[str],
    inner_validation_participants: list[str],
    all_participants: list[str],
) -> None:
    """Enforce complete participant separation between protocol roles."""

    training = set(inner_training_participants)
    validation = set(inner_validation_participants)
    outer = {str(outer_test_participant)}

    if training & validation:
        raise RuntimeError(
            "Inner training and validation participants overlap."
        )

    if training & outer:
        raise RuntimeError(
            "Outer participant entered inner training."
        )

    if validation & outer:
        raise RuntimeError(
            "Outer participant entered inner validation."
        )

    observed = training | validation | outer
    expected = set(map(str, all_participants))

    if observed != expected:
        raise RuntimeError(
            "Participant roles are not complete and disjoint."
        )


def build_model(
    model_name: str,
    config: dict[str, Any],
    *,
    random_seed: int,
) -> Any:
    """Construct one locked candidate model."""

    model_config = config["models"]

    if model_name == "extra_trees":
        parameters = model_config["extra_trees"]

        return ExtraTreesClassifier(
            n_estimators=int(
                parameters["n_estimators"]
            ),
            max_features=str(
                parameters["max_features"]
            ),
            min_samples_leaf=int(
                parameters["min_samples_leaf"]
            ),
            class_weight=str(
                parameters["class_weight"]
            ),
            n_jobs=-1,
            random_state=int(random_seed),
        )

    if model_name == "xgboost":
        parameters = model_config["xgboost"]

        return XGBClassifier(
            objective="multi:softprob",
            num_class=len(CLASS_LABELS),
            eval_metric="mlogloss",
            tree_method="hist",
            n_jobs=-1,
            random_state=int(random_seed),
            n_estimators=int(
                parameters["n_estimators"]
            ),
            max_depth=int(
                parameters["max_depth"]
            ),
            learning_rate=float(
                parameters["learning_rate"]
            ),
            subsample=float(
                parameters["subsample"]
            ),
            colsample_bytree=float(
                parameters["colsample_bytree"]
            ),
            min_child_weight=float(
                parameters["min_child_weight"]
            ),
            reg_lambda=float(
                parameters["reg_lambda"]
            ),
        )

    raise ValueError(
        f"Unsupported model: {model_name}"
    )


def predict_probabilities(
    model: Any,
    frame: pd.DataFrame,
) -> np.ndarray:
    """Return probabilities in the locked class order."""

    probabilities = np.asarray(
        model.predict_proba(frame),
        dtype=float,
    )

    model_classes = [
        int(value)
        for value in model.classes_
    ]

    expected_classes = [
        int(value)
        for value in CLASS_LABELS
    ]

    if model_classes != expected_classes:
        raise RuntimeError(
            f"Unexpected model classes: {model_classes}"
        )

    if probabilities.shape != (
        len(frame),
        len(CLASS_LABELS),
    ):
        raise RuntimeError(
            "Probability output has an unexpected shape."
        )

    if not np.isfinite(probabilities).all():
        raise RuntimeError(
            "Probability output contains nonfinite values."
        )

    if not np.allclose(
        probabilities.sum(axis=1),
        1.0,
        atol=1e-6,
    ):
        raise RuntimeError(
            "Probability rows do not sum to one."
        )

    return probabilities


def top_k_sort_value(
    mode: str,
) -> int:
    """Provide a deterministic tie-break order."""

    normalized = str(mode).lower()

    if normalized == "all":
        return 10**9

    return int(normalized)


def choose_candidate(
    candidates: pd.DataFrame,
    *,
    primary_metric: str,
    secondary_metric: str,
) -> pd.Series:
    """Choose one inner-validation candidate deterministically."""

    if candidates.empty:
        raise ValueError(
            "Candidate table is empty."
        )

    primary_column = (
        f"validation_{primary_metric}"
    )

    secondary_column = (
        f"validation_{secondary_metric}"
    )

    required = {
        primary_column,
        secondary_column,
        "selected_feature_count",
        "top_k_mode",
        "model",
        "candidate_index",
    }

    missing = required - set(
        candidates.columns
    )

    if missing:
        raise ValueError(
            f"Candidate table lacks columns: {sorted(missing)}"
        )

    ranked = candidates.copy()

    ranked["_top_k_sort"] = ranked[
        "top_k_mode"
    ].map(top_k_sort_value)

    ranked = ranked.sort_values(
        [
            primary_column,
            secondary_column,
            "selected_feature_count",
            "_top_k_sort",
            "model",
            "candidate_index",
        ],
        ascending=[
            False,
            False,
            True,
            True,
            True,
            True,
        ],
        kind="stable",
    )

    return ranked.iloc[0]


def write_json(
    path: Path,
    value: Any,
) -> None:
    path.write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )


def evaluate_outer_fold(
    table: pd.DataFrame,
    feature_names: list[str],
    config: dict[str, Any],
    *,
    outer_test_participant: str,
    outer_fold_index: int,
    fold_directory: Path,
) -> dict[str, Any]:
    """Run one completely isolated outer participant fold."""

    protocol = config["training"][
        "ds003838_nested_loso"
    ]

    random_seed = int(
        protocol["random_seed"]
    )

    participants = sorted(
        table["participant"]
        .astype(str)
        .unique()
        .tolist()
    )

    inner_training, inner_validation = (
        deterministic_inner_split(
            participants,
            outer_test_participant,
            int(
                protocol[
                    "inner_validation_participants"
                ]
            ),
        )
    )

    inner_training = list(
        map(str, inner_training)
    )

    inner_validation = list(
        map(str, inner_validation)
    )

    validate_disjoint_roles(
        outer_test_participant=outer_test_participant,
        inner_training_participants=inner_training,
        inner_validation_participants=inner_validation,
        all_participants=participants,
    )

    inner_training_frame = table.loc[
        table["participant"]
        .astype(str)
        .isin(inner_training)
    ].copy()

    inner_validation_frame = table.loc[
        table["participant"]
        .astype(str)
        .isin(inner_validation)
    ].copy()

    outer_test_frame = table.loc[
        table["participant"].astype(str)
        == str(outer_test_participant)
    ].copy()

    if len(outer_test_frame) != 108:
        raise RuntimeError(
            f"{outer_test_participant}: expected 108 test rows."
        )

    numeric_top_k = [
        int(value)
        for value in protocol[
            "candidate_top_k_modes"
        ]
        if str(value).lower() != "all"
    ]

    maximum_ranked_features = max(
        numeric_top_k
    )

    selector_parameters = config[
        "models"
    ][
        "xgboost"
    ]

    retained_features, ranked_features, inner_ranking, inner_cleanup = (
        fit_training_ranking(
            inner_training_frame,
            feature_names,
            maximum_ranked_features=
                maximum_ranked_features,
            random_seed=(
                random_seed
                + outer_fold_index * 10000
                + 101
            ),
            shap_max_samples=int(
                protocol["shap_max_samples"]
            ),
            selector_parameters=
                selector_parameters,
        )
    )

    candidate_rows: list[
        dict[str, Any]
    ] = []

    candidate_index = 0

    for top_k_mode in protocol[
        "candidate_top_k_modes"
    ]:
        selected_features = (
            selected_features_for_mode(
                str(top_k_mode),
                retained_features,
                ranked_features,
            )
        )

        for model_name in protocol[
            "candidate_models"
        ]:
            candidate_index += 1

            model = build_model(
                str(model_name),
                config,
                random_seed=(
                    random_seed
                    + outer_fold_index * 10000
                    + candidate_index
                ),
            )

            fit_classifier(
                str(model_name),
                model,
                inner_training_frame[
                    selected_features
                ],
                inner_training_frame[
                    "label"
                ].to_numpy(dtype=int),
            )

            validation_probabilities = (
                predict_probabilities(
                    model,
                    inner_validation_frame[
                        selected_features
                    ],
                )
            )

            validation_predictions = np.argmax(
                validation_probabilities,
                axis=1,
            ).astype(int)

            metrics = compute_metrics(
                inner_validation_frame[
                    "label"
                ].to_numpy(dtype=int),
                validation_predictions,
                validation_probabilities,
            )

            candidate_rows.append(
                {
                    "candidate_index":
                        candidate_index,
                    "outer_test_participant":
                        outer_test_participant,
                    "top_k_mode":
                        str(top_k_mode),
                    "selected_feature_count":
                        len(selected_features),
                    "model":
                        str(model_name),
                    **{
                        f"validation_{key}":
                            float(value)
                        for key, value
                        in metrics.items()
                    },
                }
            )

    candidates = pd.DataFrame(
        candidate_rows
    )

    chosen = choose_candidate(
        candidates,
        primary_metric=str(
            protocol[
                "primary_selection_metric"
            ]
        ),
        secondary_metric=str(
            protocol[
                "secondary_selection_metric"
            ]
        ),
    )

    chosen_top_k = str(
        chosen["top_k_mode"]
    )

    chosen_model_name = str(
        chosen["model"]
    )

    outer_training_participants = sorted(
        set(inner_training)
        | set(inner_validation)
    )

    outer_training_frame = table.loc[
        table["participant"]
        .astype(str)
        .isin(
            outer_training_participants
        )
    ].copy()

    final_retained, final_ranked, final_ranking, final_cleanup = (
        fit_training_ranking(
            outer_training_frame,
            feature_names,
            maximum_ranked_features=
                maximum_ranked_features,
            random_seed=(
                random_seed
                + outer_fold_index * 10000
                + 5001
            ),
            shap_max_samples=int(
                protocol["shap_max_samples"]
            ),
            selector_parameters=
                selector_parameters,
        )
    )

    final_selected_features = (
        selected_features_for_mode(
            chosen_top_k,
            final_retained,
            final_ranked,
        )
    )

    final_model = build_model(
        chosen_model_name,
        config,
        random_seed=(
            random_seed
            + outer_fold_index * 10000
            + 9001
        ),
    )

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

    test_probabilities = predict_probabilities(
        final_model,
        outer_test_frame[
            final_selected_features
        ],
    )

    test_predictions = np.argmax(
        test_probabilities,
        axis=1,
    ).astype(int)

    test_truth = outer_test_frame[
        "label"
    ].to_numpy(dtype=int)

    test_metrics = compute_metrics(
        test_truth,
        test_predictions,
        test_probabilities,
    )

    prediction_output = pd.DataFrame(
        {
            "segment_id":
                outer_test_frame[
                    "segment_id"
                ].astype(str).to_numpy(),
            "participant":
                outer_test_frame[
                    "participant"
                ].astype(str).to_numpy(),
            "true_label": test_truth,
            "predicted_label":
                test_predictions,
            "prob_0":
                test_probabilities[:, 0],
            "prob_1":
                test_probabilities[:, 1],
            "prob_2":
                test_probabilities[:, 2],
            "selected_top_k_mode":
                chosen_top_k,
            "selected_feature_count":
                len(final_selected_features),
            "selected_model":
                chosen_model_name,
        }
    )

    if prediction_output[
        "segment_id"
    ].duplicated().any():
        raise RuntimeError(
            "Duplicate test decisions detected."
        )

    temporary = fold_directory.with_name(
        fold_directory.name + ".building"
    )

    if temporary.exists():
        raise FileExistsError(
            f"Temporary fold exists: {temporary}"
        )

    temporary.mkdir(
        parents=True,
        exist_ok=False,
    )

    candidates.to_csv(
        temporary / "candidate_results.csv",
        index=False,
    )

    prediction_output.to_csv(
        temporary / "predictions.csv",
        index=False,
    )

    inner_ranking.to_csv(
        temporary / "inner_training_ranking.csv",
        index=False,
    )

    final_ranking.to_csv(
        temporary / "final_training_ranking.csv",
        index=False,
    )

    selection = {
        "outer_test_participant":
            outer_test_participant,
        "inner_training_participants":
            inner_training,
        "inner_validation_participants":
            inner_validation,
        "outer_training_participants":
            outer_training_participants,
        "chosen_top_k_mode":
            chosen_top_k,
        "chosen_model":
            chosen_model_name,
        "inner_validation_primary_metric":
            float(
                chosen[
                    "validation_"
                    + str(
                        protocol[
                            "primary_selection_metric"
                        ]
                    )
                ]
            ),
        "inner_validation_secondary_metric":
            float(
                chosen[
                    "validation_"
                    + str(
                        protocol[
                            "secondary_selection_metric"
                        ]
                    )
                ]
            ),
        "final_selected_feature_count":
            len(final_selected_features),
        "final_selected_features":
            final_selected_features,
        "inner_cleanup":
            inner_cleanup,
        "final_cleanup":
            final_cleanup,
        "test_metrics": {
            key: float(value)
            for key, value
            in test_metrics.items()
        },
        "probability_smoothing_applied":
            False,
    }

    write_json(
        temporary / "selection.json",
        selection,
    )

    temporary.replace(
        fold_directory
    )

    return selection


def verify_fold(
    fold_directory: Path,
    *,
    expected_participant: str,
) -> dict[str, Any]:
    """Verify one completed outer fold before reuse."""

    predictions = pd.read_csv(
        fold_directory / "predictions.csv",
        low_memory=False,
    )

    selection = json.loads(
        (
            fold_directory
            / "selection.json"
        ).read_text(encoding="utf-8")
    )

    if len(predictions) != 108:
        raise RuntimeError(
            f"{expected_participant}: expected 108 predictions."
        )

    if set(
        predictions[
            "participant"
        ].astype(str)
    ) != {
        expected_participant
    }:
        raise RuntimeError(
            "Fold predictions contain another participant."
        )

    if (
        predictions["true_label"]
        .astype(int)
        .value_counts()
        .sort_index()
        .to_dict()
        != {
            0: 36,
            1: 36,
            2: 36,
        }
    ):
        raise RuntimeError(
            f"{expected_participant}: test labels changed."
        )

    probabilities = predictions[
        [
            "prob_0",
            "prob_1",
            "prob_2",
        ]
    ].to_numpy(dtype=float)

    if not np.isfinite(
        probabilities
    ).all():
        raise RuntimeError(
            "Fold probabilities contain nonfinite values."
        )

    if not np.allclose(
        probabilities.sum(axis=1),
        1.0,
        atol=1e-6,
    ):
        raise RuntimeError(
            "Fold probability rows do not sum to one."
        )

    training = set(
        map(
            str,
            selection[
                "inner_training_participants"
            ],
        )
    )

    validation = set(
        map(
            str,
            selection[
                "inner_validation_participants"
            ],
        )
    )

    outer_training = set(
        map(
            str,
            selection[
                "outer_training_participants"
            ],
        )
    )

    if expected_participant in (
        training
        | validation
        | outer_training
    ):
        raise RuntimeError(
            "Outer participant entered a training role."
        )

    if training & validation:
        raise RuntimeError(
            "Inner training and validation roles overlap."
        )

    if outer_training != (
        training | validation
    ):
        raise RuntimeError(
            "Final outer-training registry is inconsistent."
        )

    return {
        "predictions": predictions,
        "selection": selection,
    }


def run_nested_loso(
    config: dict[str, Any],
    *,
    output_root: str | Path,
    outer_limit: int | None,
) -> dict[str, Any]:
    """Run or resume the requested outer folds."""

    protocol = config["training"][
        "ds003838_nested_loso"
    ]

    feature_root = resolve_project_path(
        "outputs/revision/features/"
        "ds003838_strict_primary"
    )

    modality = str(
        protocol["modality"]
    )

    table = pd.read_csv(
        feature_root
        / f"{modality}_features.csv",
        low_memory=False,
    )

    features = feature_columns(
        table
    )

    participants = sorted(
        table["participant"]
        .astype(str)
        .unique()
        .tolist()
    )

    if len(participants) != 58:
        raise RuntimeError(
            "Expected 58 primary participants."
        )

    requested = participants

    if outer_limit is not None:
        if outer_limit <= 0:
            raise ValueError(
                "outer_limit must be positive."
            )

        requested = participants[
            :outer_limit
        ]

    destination = resolve_project_path(
        output_root
    )

    destination.mkdir(
        parents=True,
        exist_ok=True,
    )

    folds_root = destination / "folds"

    folds_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    prediction_frames: list[
        pd.DataFrame
    ] = []

    selection_rows: list[
        dict[str, Any]
    ] = []

    total = len(requested)

    for fold_index, participant in enumerate(
        requested,
        start=1,
    ):
        print(
            f"\n[{fold_index:02d}/{total:02d}] "
            f"outer test {participant}",
            flush=True,
        )

        fold_directory = (
            folds_root / participant
        )

        if not fold_directory.exists():
            evaluate_outer_fold(
                table,
                features,
                config,
                outer_test_participant=
                    participant,
                outer_fold_index=
                    fold_index,
                fold_directory=
                    fold_directory,
            )

            print(
                "Outer fold completed.",
                flush=True,
            )

        else:
            print(
                "Existing outer fold found; "
                "verifying before skip.",
                flush=True,
            )

        verified = verify_fold(
            fold_directory,
            expected_participant=
                participant,
        )

        prediction_frames.append(
            verified["predictions"]
        )

        selection = verified[
            "selection"
        ]

        selection_rows.append(
            {
                "outer_test_participant":
                    participant,
                "chosen_top_k_mode":
                    selection[
                        "chosen_top_k_mode"
                    ],
                "chosen_model":
                    selection[
                        "chosen_model"
                    ],
                "selected_feature_count":
                    selection[
                        "final_selected_feature_count"
                    ],
                "validation_balanced_accuracy":
                    selection[
                        "inner_validation_primary_metric"
                    ],
                "validation_macro_f1":
                    selection[
                        "inner_validation_secondary_metric"
                    ],
                **{
                    f"test_{key}":
                        float(value)
                    for key, value
                    in selection[
                        "test_metrics"
                    ].items()
                },
            }
        )

        print(
            "Selected: "
            f"{selection['chosen_model']}, "
            f"top-k={selection['chosen_top_k_mode']}; "
            f"test balanced accuracy="
            f"{selection['test_metrics']['balanced_accuracy']:.6f}",
            flush=True,
        )

    predictions = pd.concat(
        prediction_frames,
        ignore_index=True,
    )

    selections = pd.DataFrame(
        selection_rows
    )

    if predictions[
        "segment_id"
    ].duplicated().any():
        raise RuntimeError(
            "Aggregated test decisions contain duplicates."
        )

    pooled_probabilities = predictions[
        [
            "prob_0",
            "prob_1",
            "prob_2",
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

    summary = {
        "dataset": "ds003838",
        "modality": modality,
        "protocol":
            "nested_leave_one_participant_out",
        "complete_primary_run":
            len(requested) == 58,
        "evaluated_outer_participant_count":
            len(requested),
        "evaluated_outer_participants":
            requested,
        "test_decision_count":
            int(len(predictions)),
        "input_feature_count":
            len(features),
        "candidate_top_k_modes":
            list(
                protocol[
                    "candidate_top_k_modes"
                ]
            ),
        "candidate_models":
            list(
                protocol[
                    "candidate_models"
                ]
            ),
        "inner_validation_participants":
            int(
                protocol[
                    "inner_validation_participants"
                ]
            ),
        "cleanup_scope":
            "inner_training_participants_only_during_selection_"
            "and_all_outer_training_participants_for_final_fit",
        "shap_scope":
            "inner_training_participants_only_during_selection_"
            "and_all_outer_training_participants_for_final_fit",
        "outer_test_participant_excluded_from":
            [
                "cleanup",
                "SHAP ranking",
                "candidate selection",
                "model fitting",
            ],
        "probability_smoothing_applied":
            False,
        "pooled_metrics": {
            key: float(value)
            for key, value
            in pooled_metrics.items()
        },
    }

    predictions.to_csv(
        destination / "predictions.csv",
        index=False,
    )

    selections.to_csv(
        destination / "fold_selections.csv",
        index=False,
    )

    write_json(
        destination / "summary.json",
        summary,
    )

    print(
        "\n===== NESTED-LOSO RUN SUMMARY ====="
    )

    print(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        )
    )

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run leakage-safe nested LOSO on "
            "ds003838 ECG+EEG features."
        )
    )

    parser.add_argument(
        "--output-root",
        default=(
            "outputs/revision/evaluation/"
            "ds003838_nested_loso"
        ),
    )

    parser.add_argument(
        "--outer-limit",
        type=int,
        default=None,
    )

    arguments = parser.parse_args()

    config = load_revision_config(
        check_input_paths=True
    )

    run_nested_loso(
        config,
        output_root=arguments.output_root,
        outer_limit=arguments.outer_limit,
    )


if __name__ == "__main__":
    main()