"""Few-shot subject-adaptive nested leave-one-participant-out evaluation.

For every outer fold:

1. One participant is excluded from global model training, feature cleanup,
   SHAP ranking, and hyperparameter selection.
2. Participant-disjoint inner validation simulates the complete few-shot
   adaptation process.
3. The held-out participant contributes only an initial labeled calibration
   period from each of the three classes.
4. A temporal guard interval separates every calibration period from its
   corresponding test period.
5. Participant-specific class prototypes are combined with a global model.
6. All remaining held-out-participant windows are evaluated once.

The protocol remains a three-class low/mid/high task. It is a labeled
few-shot subject-adaptive protocol, not calibration-free LOSO.
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
from src.evaluation.internal_personalized_final import (
    blend_probabilities,
    compute_prototype_distances,
    distances_to_probabilities,
    fit_prototype_space,
)
from src.features.internal_xr_long_window_features import (
    model_feature_columns,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_project_path(
    value: str,
) -> Path:
    """Resolve an absolute or project-relative path."""

    path = Path(value)

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path.resolve()


def normalize_by_low_state_calibration(
    frame: pd.DataFrame,
    feature_names: list[str],
    calibration_budget_seconds: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Normalize each participant using their initial low-state calibration."""

    if calibration_budget_seconds <= 0:
        raise ValueError(
            "Calibration budget must be positive."
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

    for participant, indices in transformed.groupby(
        "participant"
    ).groups.items():
        participant_indices = np.asarray(
            list(indices),
            dtype=int,
        )

        participant_frame = transformed.loc[
            participant_indices
        ]

        baseline_mask = (
            participant_frame[
                "phase"
            ].to_numpy(dtype=int)
            == 1
        ) & (
            participant_frame[
                "start_seconds"
            ].to_numpy(dtype=float)
            < calibration_budget_seconds
        )

        baseline_indices = (
            participant_indices[
                baseline_mask
            ]
        )

        if len(baseline_indices) < 1:
            raise RuntimeError(
                f"{participant}: no low-state calibration "
                f"windows for the {calibration_budget_seconds}-second budget."
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

        values = transformed.loc[
            participant_indices,
            feature_names,
        ].to_numpy(dtype=float)

        transformed.loc[
            participant_indices,
            feature_names,
        ] = (
            values
            - center
        ) / scale

        audit_rows.append(
            {
                "participant":
                    str(participant),
                "window_seconds":
                    int(
                        participant_frame[
                            "window_seconds"
                        ].iloc[0]
                    ),
                "calibration_budget_seconds":
                    calibration_budget_seconds,
                "low_state_calibration_windows":
                    int(
                        len(
                            baseline_indices
                        )
                    ),
                "low_state_calibration_raw_end_seconds":
                    float(
                        transformed.loc[
                            baseline_indices,
                            "end_seconds",
                        ].max()
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
            "Few-shot baseline normalization produced nonfinite values."
        )

    return (
        transformed,
        pd.DataFrame(
            audit_rows
        ),
    )


def split_subject_calibration_and_test(
    subject_frame: pd.DataFrame,
    *,
    calibration_budget_seconds: int,
    guard_window_multiples: int,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Create labeled calibration and later raw-disjoint test intervals."""

    if subject_frame.empty:
        raise ValueError(
            "Subject frame is empty."
        )

    if guard_window_multiples < 0:
        raise ValueError(
            "guard_window_multiples cannot be negative."
        )

    window_values = subject_frame[
        "window_seconds"
    ].astype(int).unique()

    if len(window_values) != 1:
        raise RuntimeError(
            "Subject frame contains multiple window durations."
        )

    window_seconds = int(
        window_values[0]
    )

    guard_seconds = (
        guard_window_multiples
        * window_seconds
    )

    calibration_tables: list[
        pd.DataFrame
    ] = []

    test_tables: list[
        pd.DataFrame
    ] = []

    audit_rows: list[
        dict[str, Any]
    ] = []

    participant_values = subject_frame[
        "participant"
    ].astype(str).unique()

    if len(participant_values) != 1:
        raise RuntimeError(
            "Expected exactly one participant."
        )

    participant = str(
        participant_values[0]
    )

    for phase in (
        1,
        2,
        3,
    ):
        phase_frame = subject_frame.loc[
            subject_frame[
                "phase"
            ].astype(int)
            == phase
        ].sort_values(
            [
                "start_seconds",
                "long_window_index",
            ]
        ).copy()

        if phase_frame.empty:
            raise RuntimeError(
                f"{participant}: phase {phase} is missing."
            )

        calibration = phase_frame.loc[
            phase_frame[
                "start_seconds"
            ].astype(float)
            < calibration_budget_seconds
        ].copy()

        if calibration.empty:
            raise RuntimeError(
                f"{participant}, phase {phase}: "
                "no calibration windows."
            )

        calibration_raw_end = float(
            calibration[
                "end_seconds"
            ].max()
        )

        test_start_threshold = (
            calibration_raw_end
            + guard_seconds
        )

        test = phase_frame.loc[
            phase_frame[
                "start_seconds"
            ].astype(float)
            >= test_start_threshold
        ].copy()

        if test.empty:
            raise RuntimeError(
                f"{participant}, phase {phase}: "
                "no test windows remain after calibration and guard."
            )

        if float(
            test[
                "start_seconds"
            ].min()
        ) < calibration_raw_end:
            raise RuntimeError(
                "Calibration and test raw intervals overlap."
            )

        calibration_tables.append(
            calibration
        )

        test_tables.append(
            test
        )

        audit_rows.append(
            {
                "participant":
                    participant,
                "phase":
                    phase,
                "label":
                    int(
                        phase_frame[
                            "label"
                        ].iloc[0]
                    ),
                "window_seconds":
                    window_seconds,
                "calibration_budget_seconds":
                    calibration_budget_seconds,
                "calibration_windows":
                    int(
                        len(
                            calibration
                        )
                    ),
                "calibration_raw_end_seconds":
                    calibration_raw_end,
                "guard_seconds":
                    guard_seconds,
                "test_start_threshold_seconds":
                    test_start_threshold,
                "test_windows":
                    int(
                        len(
                            test
                        )
                    ),
                "first_test_start_seconds":
                    float(
                        test[
                            "start_seconds"
                        ].min()
                    ),
            }
        )

    calibration_frame = pd.concat(
        calibration_tables,
        ignore_index=True,
    )

    test_frame = pd.concat(
        test_tables,
        ignore_index=True,
    )

    if set(
        calibration_frame[
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
            "Calibration and test segment identifiers overlap."
        )

    if set(
        calibration_frame[
            "label"
        ].astype(int).unique()
    ) != {
        0,
        1,
        2,
    }:
        raise RuntimeError(
            "Calibration does not contain all three classes."
        )

    return (
        calibration_frame,
        test_frame,
        pd.DataFrame(
            audit_rows
        ),
    )


def pooled_prototype_distances(
    calibrations: dict[str, pd.DataFrame],
    target_frame: pd.DataFrame,
    selected_features: list[str],
) -> np.ndarray:
    """Compute participant-specific prototype distances in pooled order."""

    target = target_frame.reset_index(
        drop=True
    )

    distances = np.zeros(
        (
            len(target),
            3,
        ),
        dtype=float,
    )

    for participant, indices in target.groupby(
        "participant"
    ).groups.items():
        participant_text = str(
            participant
        )

        if participant_text not in calibrations:
            raise RuntimeError(
                f"Calibration not found for {participant_text}."
            )

        ordered_indices = np.asarray(
            list(indices),
            dtype=int,
        )

        prototype_space = fit_prototype_space(
            calibrations[
                participant_text
            ],
            selected_features,
        )

        participant_distances = (
            compute_prototype_distances(
                prototype_space,
                target.loc[
                    ordered_indices
                ],
                selected_features,
            )
        )

        distances[
            ordered_indices
        ] = participant_distances

    if not np.isfinite(
        distances
    ).all():
        raise RuntimeError(
            "Prototype distance matrix contains nonfinite values."
        )

    return distances


def run_few_shot_subject_adaptive_loso(
    config: dict[str, Any],
    *,
    write_outputs: bool,
) -> dict[str, Any]:
    """Run nested few-shot adaptation for every calibration budget."""

    training_config = config[
        "training"
    ]

    protocol = training_config[
        "few_shot_subject_adaptive_loso"
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

    calibration_budgets = [
        int(value)
        for value in protocol[
            "calibration_budgets_seconds"
        ]
    ]

    guard_window_multiples = int(
        protocol[
            "guard_window_multiples"
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

    temperatures = [
        float(value)
        for value in protocol[
            "prototype_temperatures"
        ]
    ]

    model_weights = [
        float(value)
        for value in protocol[
            "global_model_probability_weights"
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

    raw_tables = load_duration_tables(
        feature_root,
        durations,
        modality,
    )

    participants = sorted(
        raw_tables[
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

    selector_parameters = training_config[
        "shap_selector"
    ]

    prepared_tables: dict[
        tuple[int, int],
        pd.DataFrame,
    ] = {}

    normalization_audits: list[
        pd.DataFrame
    ] = []

    for budget in calibration_budgets:
        for duration in durations:
            frame = raw_tables[
                duration
            ]

            features = model_feature_columns(
                frame
            )

            normalized, audit = (
                normalize_by_low_state_calibration(
                    frame,
                    features,
                    budget,
                )
            )

            prepared_tables[
                (
                    budget,
                    duration,
                )
            ] = normalized

            audit[
                "duration_seconds"
            ] = duration

            normalization_audits.append(
                audit
            )

    normalization_audit = pd.concat(
        normalization_audits,
        ignore_index=True,
    )

    all_fold_metrics: list[
        dict[str, Any]
    ] = []

    all_selections: list[
        dict[str, Any]
    ] = []

    all_validation_results: list[
        dict[str, Any]
    ] = []

    all_predictions: list[
        pd.DataFrame
    ] = []

    all_split_audits: list[
        pd.DataFrame
    ] = []

    all_rankings: list[
        pd.DataFrame
    ] = []

    all_cleanup_rows: list[
        dict[str, Any]
    ] = []

    budget_summaries: dict[
        str,
        Any,
    ] = {}

    for budget in calibration_budgets:
        print(
            f"\n############################################"
        )

        print(
            f"FEW-SHOT CALIBRATION BUDGET: {budget} SECONDS PER CLASS"
        )

        print(
            f"############################################",
            flush=True,
        )

        budget_fold_metrics: list[
            dict[str, Any]
        ] = []

        budget_predictions: list[
            pd.DataFrame
        ] = []

        budget_selections: list[
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
                f"\n===== BUDGET {budget:03d}s | "
                f"OUTER FOLD {outer_fold:02d}/{len(participants):02d} | "
                f"TEST {test_participant} =====",
                flush=True,
            )

            fold_validation_rows: list[
                dict[str, Any]
            ] = []

            for duration in durations:
                frame = prepared_tables[
                    (
                        budget,
                        duration,
                    )
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

                validation_calibrations: dict[
                    str,
                    pd.DataFrame,
                ] = {}

                validation_test_tables: list[
                    pd.DataFrame
                ] = []

                for validation_participant in (
                    inner_validation_participants
                ):
                    subject_frame = frame.loc[
                        frame[
                            "participant"
                        ].astype(str)
                        == validation_participant
                    ].copy()

                    (
                        calibration_frame,
                        test_frame,
                        split_audit,
                    ) = split_subject_calibration_and_test(
                        subject_frame,
                        calibration_budget_seconds=(
                            budget
                        ),
                        guard_window_multiples=(
                            guard_window_multiples
                        ),
                    )

                    validation_calibrations[
                        validation_participant
                    ] = calibration_frame

                    validation_test_tables.append(
                        test_frame
                    )

                    split_audit[
                        "scope"
                    ] = "inner_validation"

                    split_audit[
                        "outer_fold"
                    ] = outer_fold

                    split_audit[
                        "outer_test_participant"
                    ] = test_participant

                    split_audit[
                        "duration_seconds"
                    ] = duration

                    all_split_audits.append(
                        split_audit
                    )

                validation_test = pd.concat(
                    validation_test_tables,
                    ignore_index=True,
                )

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
                        + budget * 100000
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

                stride_values = validation_test[
                    "stride_seconds"
                ].astype(int).unique()

                if len(stride_values) != 1:
                    raise RuntimeError(
                        "Validation stride metadata is inconsistent."
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
                            + budget * 1000000
                            + outer_fold * 10000
                            + duration * 100
                            + len(
                                selected_features
                            )
                        ),
                    )

                    validation_distances = (
                        pooled_prototype_distances(
                            validation_calibrations,
                            validation_test,
                            selected_features,
                        )
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

                        model_probabilities = (
                            model.predict_proba(
                                validation_test[
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
                                            validation_test,
                                            blended,
                                            smoothing_windows,
                                        )
                                    )

                                    metrics = (
                                        evaluate_probabilities(
                                            validation_test[
                                                "label"
                                            ].to_numpy(dtype=int),
                                            smoothed,
                                        )
                                    )

                                    selection_score = float(
                                        min(
                                            metrics[
                                                "accuracy"
                                            ],
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
                                            "calibration_budget_seconds":
                                                budget,
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
                                                len(
                                                    selected_features
                                                ),
                                            "model":
                                                model_name,
                                            "prototype_temperature":
                                                temperature,
                                            "global_model_probability_weight":
                                                model_weight,
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

            all_validation_results.extend(
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
                chosen[
                    "model"
                ]
            )

            chosen_temperature = float(
                chosen[
                    "prototype_temperature"
                ]
            )

            chosen_model_weight = float(
                chosen[
                    "global_model_probability_weight"
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
                    "prototype_temperature":
                        chosen_temperature,
                    "global_model_weight":
                        chosen_model_weight,
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

            final_frame = prepared_tables[
                (
                    budget,
                    chosen_duration,
                )
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

            outer_subject_frame = final_frame.loc[
                final_frame[
                    "participant"
                ].astype(str)
                == test_participant
            ].copy()

            (
                outer_calibration,
                outer_test,
                outer_split_audit,
            ) = split_subject_calibration_and_test(
                outer_subject_frame,
                calibration_budget_seconds=(
                    budget
                ),
                guard_window_multiples=(
                    guard_window_multiples
                ),
            )

            outer_test = outer_test.reset_index(
                drop=True
            )

            outer_split_audit[
                "scope"
            ] = "outer_test"

            outer_split_audit[
                "outer_fold"
            ] = outer_fold

            outer_split_audit[
                "outer_test_participant"
            ] = test_participant

            outer_split_audit[
                "duration_seconds"
            ] = chosen_duration

            all_split_audits.append(
                outer_split_audit
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
                    + budget * 10000000
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
                    final_retained_features,
                    final_ranked_features,
                )
            )

            models = build_models(
                config,
                (
                    random_seed
                    + budget * 10000000
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

            model_probabilities = (
                final_model.predict_proba(
                    outer_test[
                        selected_features
                    ]
                )
            )

            prototype_space = fit_prototype_space(
                outer_calibration,
                selected_features,
            )

            prototype_distances = (
                compute_prototype_distances(
                    prototype_space,
                    outer_test,
                    selected_features,
                )
            )

            prototype_probabilities = (
                distances_to_probabilities(
                    prototype_distances,
                    chosen_temperature,
                )
            )

            blended = blend_probabilities(
                model_probabilities,
                prototype_probabilities,
                chosen_model_weight,
            )

            smoothed = causal_probability_smoothing(
                outer_test,
                blended,
                chosen_smoothing_windows,
            )

            labels = outer_test[
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

            fold_row = {
                "calibration_budget_seconds":
                    budget,
                "outer_fold":
                    outer_fold,
                "test_participant":
                    test_participant,
                "test_decisions":
                    int(
                        len(
                            outer_test
                        )
                    ),
                "calibration_decisions":
                    int(
                        len(
                            outer_calibration
                        )
                    ),
                "duration_seconds":
                    chosen_duration,
                "stride_seconds":
                    int(
                        outer_test[
                            "stride_seconds"
                        ].iloc[0]
                    ),
                "top_k_mode":
                    chosen_top_k_mode,
                "selected_feature_count":
                    int(
                        len(
                            selected_features
                        )
                    ),
                "model":
                    chosen_model_name,
                "prototype_temperature":
                    chosen_temperature,
                "global_model_probability_weight":
                    chosen_model_weight,
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

            budget_fold_metrics.append(
                fold_row
            )

            all_fold_metrics.append(
                fold_row
            )

            selection_row = {
                "calibration_budget_seconds":
                    budget,
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
                "prototype_temperature":
                    chosen_temperature,
                "global_model_probability_weight":
                    chosen_model_weight,
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

            budget_selections.append(
                selection_row
            )

            all_selections.append(
                selection_row
            )

            output = outer_test[
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
                "calibration_budget_seconds"
            ] = budget

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
                "selected_duration_seconds"
            ] = chosen_duration

            output[
                "selected_top_k_mode"
            ] = chosen_top_k_mode

            output[
                "selected_model"
            ] = chosen_model_name

            output[
                "selected_prototype_temperature"
            ] = chosen_temperature

            output[
                "selected_global_model_probability_weight"
            ] = chosen_model_weight

            output[
                "selected_history_seconds"
            ] = chosen_actual_history

            budget_predictions.append(
                output
            )

            all_predictions.append(
                output
            )

            ranking_output = (
                final_ranking.copy()
            )

            ranking_output[
                "calibration_budget_seconds"
            ] = budget

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

            all_rankings.append(
                ranking_output
            )

            all_cleanup_rows.append(
                {
                    "calibration_budget_seconds":
                        budget,
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
                            metrics[
                                "accuracy"
                            ],
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
                            metrics[
                                "macro_f1"
                            ],
                            4,
                        ),
                },
                flush=True,
            )

        budget_fold_table = pd.DataFrame(
            budget_fold_metrics
        ).sort_values(
            "outer_fold",
            ignore_index=True,
        )

        budget_prediction_table = pd.concat(
            budget_predictions,
            ignore_index=True,
        )

        pooled_probabilities = (
            budget_prediction_table[
                PROBABILITY_COLUMNS
            ].to_numpy(dtype=float)
        )

        pooled_labels = (
            budget_prediction_table[
                "true_label"
            ].to_numpy(dtype=int)
        )

        pooled_metrics = evaluate_probabilities(
            pooled_labels,
            pooled_probabilities,
        )

        pooled_confusion = confusion_matrix(
            pooled_labels,
            budget_prediction_table[
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
            budget_prediction_table.groupby(
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

        budget_selection_table = pd.DataFrame(
            budget_selections
        )

        budget_summaries[
            str(
                budget
            )
        ] = {
            "calibration_budget_seconds_per_class":
                budget,
            "outer_folds":
                len(participants),
            "pooled_test_decisions":
                int(
                    len(
                        budget_prediction_table
                    )
                ),
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
                        budget_fold_table[
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
                        budget_fold_table[
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
            "selection_frequency": {
                "duration_seconds":
                    budget_selection_table[
                        "duration_seconds"
                    ].value_counts().sort_index().to_dict(),
                "top_k_mode":
                    budget_selection_table[
                        "top_k_mode"
                    ].value_counts().to_dict(),
                "model":
                    budget_selection_table[
                        "model"
                    ].value_counts().to_dict(),
                "global_model_probability_weight":
                    budget_selection_table[
                        "global_model_probability_weight"
                    ].value_counts().sort_index().to_dict(),
                "actual_history_seconds":
                    budget_selection_table[
                        "actual_history_seconds"
                    ].value_counts().sort_index().to_dict(),
            },
        }

        print(
            f"\n===== BUDGET {budget}s POOLED RESULTS ====="
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
            "Phase-level accuracy:",
            f"{phase_metrics['accuracy']:.4f}",
        )

    fold_metrics = pd.DataFrame(
        all_fold_metrics
    ).sort_values(
        [
            "calibration_budget_seconds",
            "outer_fold",
        ],
        ignore_index=True,
    )

    selections = pd.DataFrame(
        all_selections
    ).sort_values(
        [
            "calibration_budget_seconds",
            "outer_fold",
        ],
        ignore_index=True,
    )

    validation_results = pd.DataFrame(
        all_validation_results
    )

    predictions = pd.concat(
        all_predictions,
        ignore_index=True,
    )

    split_audit = pd.concat(
        all_split_audits,
        ignore_index=True,
    )

    rankings = pd.concat(
        all_rankings,
        ignore_index=True,
    )

    cleanup = pd.DataFrame(
        all_cleanup_rows
    )

    best_budget = max(
        calibration_budgets,
        key=lambda value: min(
            float(
                budget_summaries[
                    str(value)
                ][
                    "pooled_metrics"
                ][
                    "accuracy"
                ]
            ),
            float(
                budget_summaries[
                    str(value)
                ][
                    "pooled_metrics"
                ][
                    "balanced_accuracy"
                ]
            ),
            float(
                budget_summaries[
                    str(value)
                ][
                    "pooled_metrics"
                ][
                    "macro_f1"
                ]
            ),
        ),
    )

    summary = {
        "experiment_type":
            "few_shot_subject_adaptive_nested_leave_one_participant_out",
        "classification_task":
            "three_class_low_mid_high",
        "calibration_free":
            False,
        "unseen_participant_global_model_training":
            True,
        "labeled_calibration_classes": [
            0,
            1,
            2,
        ],
        "calibration_budgets_seconds_per_class":
            calibration_budgets,
        "guard_window_multiples":
            guard_window_multiples,
        "participant_count":
            len(participants),
        "outer_folds":
            len(participants),
        "budgets":
            budget_summaries,
        "best_budget_by_minimum_primary_metric":
            best_budget,
        "interpretation_constraint": (
            "The held-out participant is excluded from global feature "
            "selection, configuration selection, and global model fitting. "
            "Their labeled low-, mid-, and high-state calibration windows "
            "are used for robust baseline normalization and class-prototype "
            "adaptation. This is few-shot subject-adaptive evaluation, not "
            "calibration-free LOSO."
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
                "Refusing to overwrite existing few-shot output: "
                f"{output_directory}"
            )

        if temporary_directory.exists():
            raise FileExistsError(
                "Temporary few-shot output already exists: "
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
            / "outer_predictions.csv",
            index=False,
        )

        split_audit.to_csv(
            temporary_directory
            / "calibration_test_split_audit.csv",
            index=False,
        )

        normalization_audit.to_csv(
            temporary_directory
            / "baseline_normalization_audit.csv",
            index=False,
        )

        rankings.to_csv(
            temporary_directory
            / "outer_training_shap_rankings.csv",
            index=False,
        )

        cleanup.to_csv(
            temporary_directory
            / "outer_training_cleanup.csv",
            index=False,
        )

        for budget in calibration_budgets:
            budget_directory = (
                temporary_directory
                / f"budget_{budget:03d}s"
            )

            budget_directory.mkdir(
                parents=True,
                exist_ok=False,
            )

            fold_metrics.loc[
                fold_metrics[
                    "calibration_budget_seconds"
                ]
                == budget
            ].to_csv(
                budget_directory
                / "outer_fold_metrics.csv",
                index=False,
            )

            predictions.loc[
                predictions[
                    "calibration_budget_seconds"
                ]
                == budget
            ].to_csv(
                budget_directory
                / "outer_predictions.csv",
                index=False,
            )

            (
                budget_directory
                / "budget_summary.json"
            ).write_text(
                json.dumps(
                    budget_summaries[
                        str(
                            budget
                        )
                    ],
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )

        (
            temporary_directory
            / "few_shot_summary.json"
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
        "\n===== FEW-SHOT SUBJECT-ADAPTIVE SUMMARY ====="
    )

    for budget in calibration_budgets:
        metrics = budget_summaries[
            str(
                budget
            )
        ][
            "pooled_metrics"
        ]

        print(
            f"{budget:3d}s/class | "
            f"accuracy={metrics['accuracy']:.4f} | "
            f"balanced_accuracy={metrics['balanced_accuracy']:.4f} | "
            f"macro_f1={metrics['macro_f1']:.4f} | "
            f"roc_auc={metrics['macro_roc_auc_ovr']:.4f} | "
            f"pr_auc={metrics['macro_pr_auc']:.4f}"
        )

    print(
        "Best calibration budget:",
        best_budget,
        "seconds per class",
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
            "Run nested few-shot subject-adaptive multiclass evaluation."
        )
    )

    parser.add_argument(
        "--write",
        action="store_true",
        help="Write complete few-shot evaluation artifacts.",
    )

    arguments = parser.parse_args()

    config = load_revision_config(
        check_input_paths=True
    )

    run_few_shot_subject_adaptive_loso(
        config,
        write_outputs=arguments.write,
    )


if __name__ == "__main__":
    main()