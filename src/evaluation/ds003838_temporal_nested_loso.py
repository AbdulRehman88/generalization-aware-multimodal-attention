"""Nested LOSO with inner-only ds003838 temporal-window selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.core.config import load_revision_config
from src.evaluation.ds003838_nested_loso import (
    build_model,
    predict_probabilities,
    top_k_sort_value,
    validate_disjoint_roles,
)
from src.evaluation.internal_grouped_pilot import (
    compute_metrics,
)
from src.evaluation.internal_nested_loso_long_windows import (
    deterministic_inner_split,
    fit_classifier,
    fit_training_ranking,
    selected_features_for_mode,
)
from src.features.ds003838_temporal_windows import (
    EXPECTED_WINDOW_NAMES,
    temporal_window_protocol,
)
from src.features.internal_xr_revision_features import (
    feature_columns,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

EXPECTED_PARTICIPANTS = 58
EXPECTED_ROWS_PER_PARTICIPANT = 108
EXPECTED_ROWS_PER_CLASS_PER_PARTICIPANT = 36
EXPECTED_FEATURES = 219
CLASS_LABELS = [0, 1, 2]


def resolve_project_path(
    value: str | Path,
) -> Path:
    """Resolve one project-relative path."""

    path = Path(value)

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path.resolve()


def write_json(
    path: Path,
    value: Any,
) -> None:
    """Write deterministic human-readable JSON."""

    path.write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )


def temporal_window_sort_value(
    name: str,
) -> int:
    """Return the locked prespecified temporal-window order."""

    normalized = str(name)

    if normalized not in EXPECTED_WINDOW_NAMES:
        raise ValueError(
            f"Unknown temporal window: {normalized}"
        )

    return EXPECTED_WINDOW_NAMES.index(
        normalized
    )


def choose_temporal_candidate(
    candidates: pd.DataFrame,
    *,
    primary_metric: str,
    secondary_metric: str,
) -> pd.Series:
    """Choose one joint window/top-k/model candidate deterministically."""

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
        "temporal_window",
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

    ranked["_window_sort"] = ranked[
        "temporal_window"
    ].map(
        temporal_window_sort_value
    )

    ranked = ranked.sort_values(
        [
            primary_column,
            secondary_column,
            "selected_feature_count",
            "_top_k_sort",
            "_window_sort",
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
            True,
        ],
        kind="mergesort",
    )

    return ranked.iloc[0].drop(
        labels=[
            "_top_k_sort",
            "_window_sort",
        ]
    )


def validate_temporal_tables(
    tables: dict[str, pd.DataFrame],
) -> tuple[
    list[str],
    list[str],
]:
    """Validate complete, schema-identical and trial-aligned tables."""

    if list(tables) != EXPECTED_WINDOW_NAMES:
        raise RuntimeError(
            f"Temporal table order changed: {list(tables)}"
        )

    reference = (
        tables[
            EXPECTED_WINDOW_NAMES[0]
        ]
        .sort_values("segment_id")
        .reset_index(drop=True)
    )

    reference_features = feature_columns(
        reference
    )

    if len(reference_features) != EXPECTED_FEATURES:
        raise RuntimeError(
            f"Expected {EXPECTED_FEATURES} features, "
            f"observed {len(reference_features)}."
        )

    participants = sorted(
        reference[
            "participant"
        ].astype(str).unique().tolist()
    )

    if len(participants) != EXPECTED_PARTICIPANTS:
        raise RuntimeError(
            f"Expected {EXPECTED_PARTICIPANTS} participants, "
            f"observed {len(participants)}."
        )

    for window_name in EXPECTED_WINDOW_NAMES:
        table = tables[
            window_name
        ]

        features = feature_columns(
            table
        )

        if features != reference_features:
            raise RuntimeError(
                f"{window_name}: feature schema changed."
            )

        if len(table) != (
            EXPECTED_PARTICIPANTS
            * EXPECTED_ROWS_PER_PARTICIPANT
        ):
            raise RuntimeError(
                f"{window_name}: aggregate row count changed."
            )

        if (
            table["participant"]
            .astype(str)
            .nunique()
            != EXPECTED_PARTICIPANTS
        ):
            raise RuntimeError(
                f"{window_name}: participant count changed."
            )

        if table[
            "segment_id"
        ].duplicated().any():
            raise RuntimeError(
                f"{window_name}: duplicate segments detected."
            )

        participant_counts = (
            table.groupby(
                table[
                    "participant"
                ].astype(str)
            )
            .size()
        )

        if not (
            participant_counts
            == EXPECTED_ROWS_PER_PARTICIPANT
        ).all():
            raise RuntimeError(
                f"{window_name}: participant row counts changed."
            )

        participant_class_counts = (
            table.assign(
                participant=table[
                    "participant"
                ].astype(str),
                label=table[
                    "label"
                ].astype(int),
            )
            .groupby(
                [
                    "participant",
                    "label",
                ]
            )
            .size()
            .unstack(fill_value=0)
        )

        if not (
            participant_class_counts
            == EXPECTED_ROWS_PER_CLASS_PER_PARTICIPANT
        ).all().all():
            raise RuntimeError(
                f"{window_name}: participant-class balance changed."
            )

        values = table[
            features
        ].to_numpy(dtype=float)

        if not np.isfinite(
            values
        ).all():
            raise RuntimeError(
                f"{window_name}: nonfinite features detected."
            )

        candidate = (
            table.sort_values("segment_id")
            .reset_index(drop=True)
        )

        for column in [
            "segment_id",
            "participant",
            "phase",
            "label",
            "window_index",
        ]:
            if not reference[
                column
            ].astype(str).equals(
                candidate[
                    column
                ].astype(str)
            ):
                raise RuntimeError(
                    f"{window_name}: metadata differs for {column}."
                )

    return reference_features, participants


def load_temporal_tables(
    feature_root: str | Path,
) -> tuple[
    dict[str, pd.DataFrame],
    list[str],
    list[str],
]:
    """Load all locked temporal candidate aggregates."""

    root = resolve_project_path(
        feature_root
    )

    tables: dict[
        str,
        pd.DataFrame
    ] = {}

    for window_name in EXPECTED_WINDOW_NAMES:
        path = (
            root
            / window_name
            / "ECG_EEG_delta_features.csv"
        )

        if not path.is_file():
            raise FileNotFoundError(
                f"Temporal feature table missing: {path}"
            )

        tables[
            window_name
        ] = pd.read_csv(
            path,
            low_memory=False,
        )

    features, participants = (
        validate_temporal_tables(
            tables
        )
    )

    return tables, features, participants


def role_frame(
    table: pd.DataFrame,
    participants: list[str],
) -> pd.DataFrame:
    """Return one participant-restricted frame."""

    return table.loc[
        table[
            "participant"
        ].astype(str).isin(
            list(
                map(
                    str,
                    participants,
                )
            )
        )
    ].copy()


def evaluate_outer_fold(
    tables: dict[str, pd.DataFrame],
    feature_names: list[str],
    config: dict[str, Any],
    *,
    outer_test_participant: str,
    outer_fold_index: int,
    fold_directory: Path,
) -> dict[str, Any]:
    """Run one outer fold with inner-only temporal selection."""

    protocol = config[
        "training"
    ][
        "ds003838_nested_loso"
    ]

    temporal_protocol = temporal_window_protocol(
        config
    )

    configured_windows = [
        str(candidate["name"])
        for candidate in temporal_protocol[
            "candidates"
        ]
    ]

    if configured_windows != EXPECTED_WINDOW_NAMES:
        raise RuntimeError(
            "Configured temporal-window order changed."
        )

    random_seed = int(
        protocol["random_seed"]
    )

    participants = sorted(
        tables[
            EXPECTED_WINDOW_NAMES[0]
        ][
            "participant"
        ].astype(str).unique().tolist()
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
        map(
            str,
            inner_training,
        )
    )

    inner_validation = list(
        map(
            str,
            inner_validation,
        )
    )

    validate_disjoint_roles(
        outer_test_participant=
            outer_test_participant,
        inner_training_participants=
            inner_training,
        inner_validation_participants=
            inner_validation,
        all_participants=
            participants,
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

    candidate_rows: list[
        dict[str, Any]
    ] = []

    inner_rankings: dict[
        str,
        pd.DataFrame
    ] = {}

    inner_cleanup_by_window: dict[
        str,
        Any
    ] = {}

    candidate_index = 0

    for window_index, window_name in enumerate(
        EXPECTED_WINDOW_NAMES,
        start=1,
    ):
        table = tables[
            window_name
        ]

        inner_training_frame = role_frame(
            table,
            inner_training,
        )

        inner_validation_frame = role_frame(
            table,
            inner_validation,
        )

        retained_features, ranked_features, ranking, cleanup = (
            fit_training_ranking(
                inner_training_frame,
                feature_names,
                maximum_ranked_features=
                    maximum_ranked_features,
                random_seed=(
                    random_seed
                    + outer_fold_index * 100000
                    + window_index * 10000
                    + 101
                ),
                shap_max_samples=int(
                    protocol[
                        "shap_max_samples"
                    ]
                ),
                selector_parameters=
                    selector_parameters,
            )
        )

        inner_rankings[
            window_name
        ] = ranking

        inner_cleanup_by_window[
            window_name
        ] = cleanup

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
                        + outer_fold_index * 100000
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
                        "temporal_window":
                            window_name,
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

    expected_candidate_count = (
        len(EXPECTED_WINDOW_NAMES)
        * len(
            protocol[
                "candidate_top_k_modes"
            ]
        )
        * len(
            protocol[
                "candidate_models"
            ]
        )
    )

    if len(candidates) != expected_candidate_count:
        raise RuntimeError(
            f"Expected {expected_candidate_count} candidates, "
            f"observed {len(candidates)}."
        )

    chosen = choose_temporal_candidate(
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

    chosen_window = str(
        chosen[
            "temporal_window"
        ]
    )

    chosen_top_k = str(
        chosen[
            "top_k_mode"
        ]
    )

    chosen_model_name = str(
        chosen[
            "model"
        ]
    )

    outer_training_participants = sorted(
        set(inner_training)
        | set(inner_validation)
    )

    chosen_table = tables[
        chosen_window
    ]

    outer_training_frame = role_frame(
        chosen_table,
        outer_training_participants,
    )

    outer_test_frame = chosen_table.loc[
        chosen_table[
            "participant"
        ].astype(str)
        == str(
            outer_test_participant
        )
    ].copy()

    if len(outer_test_frame) != EXPECTED_ROWS_PER_PARTICIPANT:
        raise RuntimeError(
            f"{outer_test_participant}: expected "
            f"{EXPECTED_ROWS_PER_PARTICIPANT} test rows."
        )

    (
        final_retained,
        final_ranked,
        final_ranking,
        final_cleanup,
    ) = fit_training_ranking(
        outer_training_frame,
        feature_names,
        maximum_ranked_features=
            maximum_ranked_features,
        random_seed=(
            random_seed
            + outer_fold_index * 100000
            + 50001
        ),
        shap_max_samples=int(
            protocol[
                "shap_max_samples"
            ]
        ),
        selector_parameters=
            selector_parameters,
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
            + outer_fold_index * 100000
            + 90001
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

    test_metrics = compute_metrics(
        outer_test_frame[
            "label"
        ].to_numpy(dtype=int),
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
            "true_label":
                outer_test_frame[
                    "label"
                ].to_numpy(dtype=int),
            "predicted_label":
                test_predictions,
            "prob_0":
                test_probabilities[:, 0],
            "prob_1":
                test_probabilities[:, 1],
            "prob_2":
                test_probabilities[:, 2],
            "selected_temporal_window":
                chosen_window,
            "selected_top_k_mode":
                chosen_top_k,
            "selected_feature_count":
                len(
                    final_selected_features
                ),
            "selected_model":
                chosen_model_name,
        }
    )

    if prediction_output[
        "segment_id"
    ].duplicated().any():
        raise RuntimeError(
            "Duplicate outer-test decisions detected."
        )

    temporary = fold_directory.with_name(
        fold_directory.name + ".building"
    )

    if temporary.exists():
        raise FileExistsError(
            f"Temporary fold already exists: {temporary}"
        )

    temporary.mkdir(
        parents=True,
        exist_ok=False,
    )

    rankings_directory = (
        temporary
        / "inner_training_rankings"
    )

    rankings_directory.mkdir(
        parents=True,
        exist_ok=False,
    )

    candidates.to_csv(
        temporary
        / "candidate_results.csv",
        index=False,
    )

    prediction_output.to_csv(
        temporary
        / "predictions.csv",
        index=False,
    )

    for window_name in EXPECTED_WINDOW_NAMES:
        inner_rankings[
            window_name
        ].to_csv(
            rankings_directory
            / f"{window_name}.csv",
            index=False,
        )

    final_ranking.to_csv(
        temporary
        / "final_training_ranking.csv",
        index=False,
    )

    selection = {
        "outer_test_participant":
            outer_test_participant,
        "all_participants":
            participants,
        "inner_training_participants":
            inner_training,
        "inner_validation_participants":
            inner_validation,
        "outer_training_participants":
            outer_training_participants,
        "candidate_temporal_windows":
            EXPECTED_WINDOW_NAMES,
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
        "candidate_count":
            int(len(candidates)),
        "chosen_temporal_window":
            chosen_window,
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
            len(
                final_selected_features
            ),
        "final_selected_features":
            final_selected_features,
        "inner_cleanup_by_window":
            inner_cleanup_by_window,
        "final_cleanup":
            final_cleanup,
        "selection_scope":
            "inner_training_and_inner_validation_participants_only",
        "outer_test_participant_excluded_from": [
            "temporal-window selection",
            "feature cleanup",
            "SHAP ranking",
            "top-k selection",
            "model selection",
            "model fitting",
        ],
        "test_metrics": {
            key: float(value)
            for key, value
            in test_metrics.items()
        },
    }

    write_json(
        temporary
        / "selection.json",
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
    expected_candidate_count: int,
) -> dict[str, Any]:
    """Verify one completed temporal-selection fold."""

    required = [
        fold_directory
        / "candidate_results.csv",
        fold_directory
        / "predictions.csv",
        fold_directory
        / "selection.json",
        fold_directory
        / "final_training_ranking.csv",
    ]

    missing = [
        str(path)
        for path in required
        if not path.is_file()
    ]

    if missing:
        raise RuntimeError(
            f"{expected_participant}: incomplete fold: {missing}"
        )

    candidates = pd.read_csv(
        fold_directory
        / "candidate_results.csv",
        low_memory=False,
    )

    predictions = pd.read_csv(
        fold_directory
        / "predictions.csv",
        low_memory=False,
    )

    selection = json.loads(
        (
            fold_directory
            / "selection.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    if len(candidates) != expected_candidate_count:
        raise RuntimeError(
            f"{expected_participant}: expected "
            f"{expected_candidate_count} candidates, "
            f"observed {len(candidates)}."
        )

    if set(
        candidates[
            "temporal_window"
        ].astype(str)
    ) != set(
        EXPECTED_WINDOW_NAMES
    ):
        raise RuntimeError(
            f"{expected_participant}: window candidates changed."
        )

    candidates_per_window = (
        candidates.groupby(
            "temporal_window"
        ).size()
    )

    if not (
        candidates_per_window
        == (
            expected_candidate_count
            // len(
                EXPECTED_WINDOW_NAMES
            )
        )
    ).all():
        raise RuntimeError(
            f"{expected_participant}: candidates are unbalanced "
            "across temporal windows."
        )

    if len(predictions) != EXPECTED_ROWS_PER_PARTICIPANT:
        raise RuntimeError(
            f"{expected_participant}: expected "
            f"{EXPECTED_ROWS_PER_PARTICIPANT} predictions."
        )

    if set(
        predictions[
            "participant"
        ].astype(str)
    ) != {
        expected_participant
    }:
        raise RuntimeError(
            f"{expected_participant}: fold contains another participant."
        )

    if (
        predictions[
            "true_label"
        ].astype(int)
        .value_counts()
        .sort_index()
        .to_dict()
        != {
            0:
                EXPECTED_ROWS_PER_CLASS_PER_PARTICIPANT,
            1:
                EXPECTED_ROWS_PER_CLASS_PER_PARTICIPANT,
            2:
                EXPECTED_ROWS_PER_CLASS_PER_PARTICIPANT,
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
            f"{expected_participant}: nonfinite probabilities."
        )

    if not np.allclose(
        probabilities.sum(axis=1),
        1.0,
        atol=1e-6,
    ):
        raise RuntimeError(
            f"{expected_participant}: probabilities do not sum to one."
        )

    chosen_window = str(
        selection[
            "chosen_temporal_window"
        ]
    )

    if chosen_window not in EXPECTED_WINDOW_NAMES:
        raise RuntimeError(
            f"{expected_participant}: invalid selected window."
        )

    if set(
        predictions[
            "selected_temporal_window"
        ].astype(str)
    ) != {
        chosen_window
    }:
        raise RuntimeError(
            f"{expected_participant}: prediction window mismatch."
        )

    if set(
        predictions[
            "selected_top_k_mode"
        ].astype(str)
    ) != {
        str(
            selection[
                "chosen_top_k_mode"
            ]
        )
    }:
        raise RuntimeError(
            f"{expected_participant}: prediction top-k mismatch."
        )

    if set(
        predictions[
            "selected_model"
        ].astype(str)
    ) != {
        str(
            selection[
                "chosen_model"
            ]
        )
    }:
        raise RuntimeError(
            f"{expected_participant}: prediction model mismatch."
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

    outer = {
        expected_participant
    }

    if training & validation:
        raise RuntimeError(
            f"{expected_participant}: inner roles overlap."
        )

    if outer & training:
        raise RuntimeError(
            f"{expected_participant}: outer participant entered training."
        )

    if outer & validation:
        raise RuntimeError(
            f"{expected_participant}: outer participant entered validation."
        )

    if outer & outer_training:
        raise RuntimeError(
            f"{expected_participant}: outer participant entered final fitting."
        )

    if outer_training != (
        training
        | validation
    ):
        raise RuntimeError(
            f"{expected_participant}: outer-training registry changed."
        )

    return {
        "predictions":
            predictions,
        "selection":
            selection,
        "candidates":
            candidates,
    }


def participant_metric_table(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Compute one complete metric vector per outer participant."""

    rows = []

    for participant, frame in predictions.groupby(
        "participant",
        sort=True,
    ):
        probabilities = frame[
            [
                "prob_0",
                "prob_1",
                "prob_2",
            ]
        ].to_numpy(dtype=float)

        metrics = compute_metrics(
            frame[
                "true_label"
            ].to_numpy(dtype=int),
            frame[
                "predicted_label"
            ].to_numpy(dtype=int),
            probabilities,
        )

        rows.append(
            {
                "participant":
                    str(participant),
                **{
                    key: float(value)
                    for key, value
                    in metrics.items()
                },
            }
        )

    return pd.DataFrame(
        rows
    )


def participant_bootstrap_ci(
    participant_metrics: pd.DataFrame,
    *,
    repetitions: int,
    random_seed: int,
) -> pd.DataFrame:
    """Bootstrap participant-mean metrics with participant as the unit."""

    metric_columns = [
        column
        for column in participant_metrics.columns
        if column != "participant"
    ]

    values = participant_metrics[
        metric_columns
    ].to_numpy(dtype=float)

    if len(values) == 0:
        raise ValueError(
            "No participant metrics available."
        )

    generator = np.random.default_rng(
        int(random_seed)
    )

    bootstrap_means = np.empty(
        (
            int(repetitions),
            len(metric_columns),
        ),
        dtype=float,
    )

    for repetition in range(
        int(repetitions)
    ):
        indices = generator.integers(
            0,
            len(values),
            size=len(values),
        )

        bootstrap_means[
            repetition
        ] = np.mean(
            values[
                indices
            ],
            axis=0,
        )

    rows = []

    for metric_index, metric in enumerate(
        metric_columns
    ):
        distribution = bootstrap_means[
            :,
            metric_index,
        ]

        rows.append(
            {
                "metric": metric,
                "participant_mean":
                    float(
                        np.mean(
                            values[
                                :,
                                metric_index,
                            ]
                        )
                    ),
                "bootstrap_mean":
                    float(
                        np.mean(
                            distribution
                        )
                    ),
                "ci_2_5":
                    float(
                        np.quantile(
                            distribution,
                            0.025,
                        )
                    ),
                "ci_97_5":
                    float(
                        np.quantile(
                            distribution,
                            0.975,
                        )
                    ),
                "bootstrap_repetitions":
                    int(repetitions),
            }
        )

    return pd.DataFrame(
        rows
    )


def run_temporal_nested_loso(
    config: dict[str, Any],
    *,
    feature_root: str | Path,
    output_root: str | Path,
    outer_limit: int | None,
) -> dict[str, Any]:
    """Run or resume temporal-window nested LOSO."""

    protocol = config[
        "training"
    ][
        "ds003838_nested_loso"
    ]

    temporal_window_protocol(
        config
    )

    tables, features, participants = (
        load_temporal_tables(
            feature_root
        )
    )

    if outer_limit is None:
        requested = participants
    else:
        if int(outer_limit) <= 0:
            raise ValueError(
                "outer_limit must be positive."
            )

        requested = participants[
            : int(outer_limit)
        ]

    destination = resolve_project_path(
        output_root
    )

    destination.mkdir(
        parents=True,
        exist_ok=True,
    )

    folds_root = (
        destination
        / "folds"
    )

    folds_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    expected_candidate_count = (
        len(EXPECTED_WINDOW_NAMES)
        * len(
            protocol[
                "candidate_top_k_modes"
            ]
        )
        * len(
            protocol[
                "candidate_models"
            ]
        )
    )

    prediction_frames: list[
        pd.DataFrame
    ] = []

    selection_rows: list[
        dict[str, Any]
    ] = []

    total = len(
        requested
    )

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
            folds_root
            / participant
        )

        if not fold_directory.exists():
            evaluate_outer_fold(
                tables,
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
                "Temporal-selection fold completed.",
                flush=True,
            )
        else:
            print(
                "Existing fold found; verifying before skip.",
                flush=True,
            )

        verified = verify_fold(
            fold_directory,
            expected_participant=
                participant,
            expected_candidate_count=
                expected_candidate_count,
        )

        predictions = verified[
            "predictions"
        ]

        selection = verified[
            "selection"
        ]

        prediction_frames.append(
            predictions
        )

        selection_rows.append(
            {
                "outer_test_participant":
                    participant,
                "chosen_temporal_window":
                    selection[
                        "chosen_temporal_window"
                    ],
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
            f"window={selection['chosen_temporal_window']}, "
            f"model={selection['chosen_model']}, "
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

    probabilities = predictions[
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
        probabilities,
    )

    participant_metrics = (
        participant_metric_table(
            predictions
        )
    )

    bootstrap = participant_bootstrap_ci(
        participant_metrics,
        repetitions=int(
            protocol[
                "bootstrap_repetitions"
            ]
        ),
        random_seed=int(
            protocol[
                "random_seed"
            ]
        ) + 770001,
    )

    predictions.to_csv(
        destination
        / "predictions.csv",
        index=False,
    )

    selections.to_csv(
        destination
        / "fold_selections.csv",
        index=False,
    )

    participant_metrics.to_csv(
        destination
        / "participant_metrics.csv",
        index=False,
    )

    bootstrap.to_csv(
        destination
        / "participant_bootstrap_ci.csv",
        index=False,
    )

    summary = {
        "dataset": "ds003838",
        "cohort": "strict_primary",
        "evaluation_protocol":
            "nested_leave_one_participant_out_with_inner_temporal_selection",
        "complete_primary_run":
            len(requested)
            == EXPECTED_PARTICIPANTS,
        "evaluated_outer_participant_count":
            len(requested),
        "evaluated_outer_participants":
            requested,
        "test_decision_count":
            int(
                len(
                    predictions
                )
            ),
        "input_feature_count_per_window":
            len(features),
        "candidate_temporal_windows":
            EXPECTED_WINDOW_NAMES,
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
        "candidate_count_per_fold":
            expected_candidate_count,
        "inner_validation_participants":
            int(
                protocol[
                    "inner_validation_participants"
                ]
            ),
        "temporal_window_selection_scope":
            "participant_disjoint_inner_validation_only",
        "cleanup_scope":
            "window_specific_inner_training_only_during_selection_"
            "and_selected_window_outer_training_only_for_final_fit",
        "shap_scope":
            "window_specific_inner_training_only_during_selection_"
            "and_selected_window_outer_training_only_for_final_fit",
        "outer_test_participant_excluded_from": [
            "temporal-window selection",
            "feature cleanup",
            "SHAP ranking",
            "top-k selection",
            "model selection",
            "model fitting",
        ],
        "probability_smoothing_applied":
            False,
        "pooled_metrics": {
            key: float(value)
            for key, value
            in pooled_metrics.items()
        },
        "selected_window_counts": {
            str(key): int(value)
            for key, value
            in selections[
                "chosen_temporal_window"
            ]
            .value_counts()
            .sort_index()
            .to_dict()
            .items()
        },
        "selected_model_counts": {
            str(key): int(value)
            for key, value
            in selections[
                "chosen_model"
            ]
            .value_counts()
            .sort_index()
            .to_dict()
            .items()
        },
        "selected_top_k_counts": {
            str(key): int(value)
            for key, value
            in selections[
                "chosen_top_k_mode"
            ]
            .astype(str)
            .value_counts()
            .sort_index()
            .to_dict()
            .items()
        },
    }

    write_json(
        destination
        / "summary.json",
        summary,
    )

    print(
        "\n===== TEMPORAL NESTED-LOSO SUMMARY ====="
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
            "Run ds003838 nested LOSO with "
            "inner-only temporal-window selection."
        )
    )

    parser.add_argument(
        "--feature-root",
        default=(
            "outputs/revision/features/"
            "ds003838_temporal_candidates"
        ),
    )

    parser.add_argument(
        "--output-root",
        default=(
            "outputs/revision/evaluation/"
            "ds003838_temporal_nested_loso"
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

    run_temporal_nested_loso(
        config,
        feature_root=
            arguments.feature_root,
        output_root=
            arguments.output_root,
        outer_limit=
            arguments.outer_limit,
    )


if __name__ == "__main__":
    main()