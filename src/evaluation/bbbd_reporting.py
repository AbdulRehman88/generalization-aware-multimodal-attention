"""Fold-aware reporting utilities for the locked BBBD evaluation.

Threshold-dependent metrics use each recording's stored validation-selected
decision. Probability-ranking metrics use the original out-of-fold or
cross-experiment probabilities without threshold shifting.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)

from src.evaluation.bbbd_protocol import (
    bootstrap_confidence_intervals,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

REQUIRED_RECORDING_COLUMNS = {
    "recording_id",
    "participant",
    "label",
    "probability",
    "threshold",
    "predicted_label",
}


def resolve_project_path(
    value: str | Path,
) -> Path:
    """Resolve a repository-relative path."""

    path = Path(value)

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path.resolve()


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
    """Write a CSV table atomically."""

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


def validate_fold_aware_recording_predictions(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Validate stored recording probabilities, thresholds, and decisions."""

    missing = (
        REQUIRED_RECORDING_COLUMNS
        - set(
            frame.columns
        )
    )

    if missing:
        raise ValueError(
            "Recording predictions lack columns: "
            f"{sorted(missing)}"
        )

    if frame.empty:
        raise ValueError(
            "Recording-prediction table is empty."
        )

    if frame[
        "recording_id"
    ].astype(str).duplicated().any():
        raise ValueError(
            "Duplicate recording IDs were detected."
        )

    result = frame.copy()

    result[
        "label"
    ] = pd.to_numeric(
        result[
            "label"
        ],
        errors="raise",
    ).astype(int)

    result[
        "predicted_label"
    ] = pd.to_numeric(
        result[
            "predicted_label"
        ],
        errors="raise",
    ).astype(int)

    result[
        "probability"
    ] = pd.to_numeric(
        result[
            "probability"
        ],
        errors="coerce",
    ).astype(float)

    result[
        "threshold"
    ] = pd.to_numeric(
        result[
            "threshold"
        ],
        errors="coerce",
    ).astype(float)

    if not set(
        result[
            "label"
        ].unique()
    ).issubset(
        {
            0,
            1,
        }
    ):
        raise ValueError(
            "True recording labels must be binary."
        )

    if not set(
        result[
            "predicted_label"
        ].unique()
    ).issubset(
        {
            0,
            1,
        }
    ):
        raise ValueError(
            "Stored recording decisions must be binary."
        )

    for column in [
        "probability",
        "threshold",
    ]:
        values = result[
            column
        ].to_numpy(
            dtype=float
        )

        if not np.isfinite(
            values
        ).all():
            raise ValueError(
                f"{column} contains nonfinite values."
            )

        if np.any(
            (
                values < 0.0
            )
            | (
                values > 1.0
            )
        ):
            raise ValueError(
                f"{column} contains values outside [0, 1]."
            )

    reconstructed = (
        result[
            "probability"
        ].to_numpy(
            dtype=float
        )
        >= result[
            "threshold"
        ].to_numpy(
            dtype=float
        )
    ).astype(int)

    stored = result[
        "predicted_label"
    ].to_numpy(
        dtype=int
    )

    if not np.array_equal(
        reconstructed,
        stored,
    ):
        mismatch = result.loc[
            reconstructed
            != stored,
            [
                "recording_id",
                "probability",
                "threshold",
                "predicted_label",
            ],
        ]

        raise ValueError(
            "Stored recording decisions do not match their "
            "fold-specific thresholds:\n"
            + mismatch.head(
                20
            ).to_string(
                index=False
            )
        )

    return result


def fold_aware_recording_metrics(
    frame: pd.DataFrame,
) -> dict[str, float]:
    """Compute metrics using stored decisions and untouched probabilities."""

    validated = validate_fold_aware_recording_predictions(
        frame
    )

    labels = validated[
        "label"
    ].to_numpy(
        dtype=int
    )

    predictions = validated[
        "predicted_label"
    ].to_numpy(
        dtype=int
    )

    probabilities = validated[
        "probability"
    ].to_numpy(
        dtype=float
    )

    unique_labels = np.unique(
        labels
    )

    if len(
        unique_labels
    ) == 2:
        roc_auc = float(
            roc_auc_score(
                labels,
                probabilities,
            )
        )

        pr_auc = float(
            average_precision_score(
                labels,
                probabilities,
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
                    labels,
                    predictions,
                )
            ),
        "balanced_accuracy":
            float(
                balanced_accuracy_score(
                    labels,
                    predictions,
                )
            ),
        "macro_f1":
            float(
                f1_score(
                    labels,
                    predictions,
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


def fold_aware_participant_metric_table(
    frame: pd.DataFrame,
    *,
    participant_column: str = "participant",
) -> pd.DataFrame:
    """Compute participant-specific recording metrics."""

    if participant_column not in frame.columns:
        raise ValueError(
            f"Participant column is missing: {participant_column}"
        )

    rows = []

    for participant, participant_frame in frame.groupby(
        participant_column,
        sort=True,
    ):
        metrics = fold_aware_recording_metrics(
            participant_frame
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
                            participant_frame
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


def fold_aware_participant_macro_metrics(
    frame: pd.DataFrame,
    *,
    participant_column: str = "participant",
) -> dict[str, float]:
    """Macro-average participant-specific recording metrics."""

    table = fold_aware_participant_metric_table(
        frame,
        participant_column=
            participant_column,
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


def fold_aware_participant_cluster_bootstrap(
    frame: pd.DataFrame,
    *,
    repetitions: int,
    random_seed: int,
) -> pd.DataFrame:
    """Bootstrap participants while preserving stored fold decisions."""

    validated = validate_fold_aware_recording_predictions(
        frame
    )

    participants = sorted(
        validated[
            "participant"
        ].astype(str).unique()
    )

    if not participants:
        raise ValueError(
            "No participants are available for bootstrap."
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
            validated.loc[
                validated[
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
            participant = str(
                participant
            )

            sampled_frame = participant_frames[
                participant
            ].copy()

            bootstrap_participant = (
                f"draw-{draw_index:05d}::"
                f"{participant}"
            )

            sampled_frame[
                "_bootstrap_participant"
            ] = bootstrap_participant

            sampled_frame[
                "_source_recording_id"
            ] = sampled_frame[
                "recording_id"
            ].astype(str)

            sampled_frame[
                "recording_id"
            ] = (
                bootstrap_participant
                + "::"
                + sampled_frame[
                    "_source_recording_id"
                ].astype(str)
            )

            sampled_frames.append(
                sampled_frame
            )

        bootstrap_frame = pd.concat(
            sampled_frames,
            ignore_index=True,
        )

        pooled = fold_aware_recording_metrics(
            bootstrap_frame
        )

        participant_macro = (
            fold_aware_participant_macro_metrics(
                bootstrap_frame,
                participant_column=
                    "_bootstrap_participant",
            )
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


def combine_split_recording_predictions(
    split_directories: Sequence[str | Path],
) -> pd.DataFrame:
    """Combine verified completed split predictions."""

    if not split_directories:
        raise ValueError(
            "At least one split directory is required."
        )

    frames = []

    for split_directory in split_directories:
        directory = resolve_project_path(
            split_directory
        )

        completion_path = (
            directory
            / "complete.json"
        )

        prediction_path = (
            directory
            / "recording_predictions.csv"
        )

        if not completion_path.is_file():
            raise FileNotFoundError(
                completion_path
            )

        if not prediction_path.is_file():
            raise FileNotFoundError(
                prediction_path
            )

        completion = json.loads(
            completion_path.read_text(
                encoding="utf-8"
            )
        )

        if not bool(
            completion.get(
                "completed",
                False,
            )
        ):
            raise RuntimeError(
                f"Split is not complete: {directory}"
            )

        frame = pd.read_csv(
            prediction_path,
            low_memory=False,
        )

        frame = validate_fold_aware_recording_predictions(
            frame
        )

        unique_thresholds = sorted(
            frame[
                "threshold"
            ].astype(float).unique().tolist()
        )

        if len(
            unique_thresholds
        ) != 1:
            raise RuntimeError(
                "One completed split contains multiple thresholds."
            )

        stored_threshold = float(
            completion[
                "selected_threshold"
            ]
        )

        if not np.isclose(
            unique_thresholds[
                0
            ],
            stored_threshold,
            atol=1e-15,
            rtol=0.0,
        ):
            raise RuntimeError(
                "Prediction threshold differs from complete.json."
            )

        frame[
            "split_directory"
        ] = directory.as_posix()

        frame[
            "split_sha256"
        ] = str(
            completion[
                "split_sha256"
            ]
        )

        frames.append(
            frame
        )

    combined = pd.concat(
        frames,
        ignore_index=True,
    )

    if combined[
        "recording_id"
    ].astype(str).duplicated().any():
        duplicates = combined.loc[
            combined[
                "recording_id"
            ].astype(str).duplicated(
                keep=False
            ),
            [
                "recording_id",
                "split_directory",
            ],
        ]

        raise RuntimeError(
            "Combined split predictions contain duplicate recordings:\n"
            + duplicates.head(
                30
            ).to_string(
                index=False
            )
        )

    return validate_fold_aware_recording_predictions(
        combined
    )


def write_fold_aware_report(
    *,
    recording_predictions: pd.DataFrame,
    output_directory: str | Path,
    protocol_identity: str,
    bootstrap_repetitions: int,
    bootstrap_seed: int,
    confidence_level: float,
) -> dict[str, Any]:
    """Write verified fold-aware metrics and participant confidence intervals."""

    validated = validate_fold_aware_recording_predictions(
        recording_predictions
    )

    output = resolve_project_path(
        output_directory
    )

    if output.exists() and any(
        output.iterdir()
    ):
        raise RuntimeError(
            f"Reporting output is nonempty: {output}"
        )

    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    pooled = fold_aware_recording_metrics(
        validated
    )

    participant_table = (
        fold_aware_participant_metric_table(
            validated
        )
    )

    participant_macro = (
        fold_aware_participant_macro_metrics(
            validated
        )
    )

    bootstrap = (
        fold_aware_participant_cluster_bootstrap(
            validated,
            repetitions=
                int(
                    bootstrap_repetitions
                ),
            random_seed=
                int(
                    bootstrap_seed
                ),
        )
    )

    confidence = bootstrap_confidence_intervals(
        bootstrap,
        confidence_level=
            float(
                confidence_level
            ),
    )

    write_csv_atomic(
        output
        / "recording_predictions.csv",
        validated,
    )

    write_csv_atomic(
        output
        / "participant_metrics.csv",
        participant_table,
    )

    write_csv_atomic(
        output
        / "participant_bootstrap.csv",
        bootstrap,
    )

    write_csv_atomic(
        output
        / "confidence_intervals.csv",
        confidence,
    )

    threshold_counts = (
        validated[
            "threshold"
        ]
        .astype(float)
        .value_counts()
        .sort_index()
        .to_dict()
    )

    summary = {
        "protocol_identity":
            str(
                protocol_identity
            ),
        "participants":
            int(
                validated[
                    "participant"
                ].nunique()
            ),
        "recordings":
            int(
                len(
                    validated
                )
            ),
        "recording_label_counts": {
            str(
                int(
                    key
                )
            ):
                int(
                    value
                )
            for key, value
            in validated[
                "label"
            ]
            .astype(int)
            .value_counts()
            .sort_index()
            .to_dict()
            .items()
        },
        "threshold_counts": {
            str(
                float(
                    key
                )
            ):
                int(
                    value
                )
            for key, value
            in threshold_counts.items()
        },
        "pooled_recording_metrics":
            pooled,
        "participant_macro_metrics":
            participant_macro,
        "bootstrap_unit":
            "participant",
        "bootstrap_repetitions":
            int(
                bootstrap_repetitions
            ),
        "confidence_level":
            float(
                confidence_level
            ),
        "threshold_dependent_metric_policy":
            (
                "Use each recording's stored validation-selected "
                "fold or transfer threshold."
            ),
        "probability_metric_policy":
            (
                "Use untouched out-of-fold or target-experiment "
                "probabilities without threshold shifting."
            ),
    }

    write_json_atomic(
        output
        / "summary.json",
        summary,
    )

    return summary


def synthetic_fold_aware_predictions() -> pd.DataFrame:
    """Create predictions where a single global threshold is demonstrably wrong."""

    rows = []

    specifications = [
        (
            "p1",
            0.30,
            0.20,
            0.40,
        ),
        (
            "p2",
            0.70,
            0.35,
            0.80,
        ),
        (
            "p3",
            0.50,
            0.30,
            0.70,
        ),
    ]

    for participant, threshold, negative_probability, positive_probability in specifications:
        for label, probability in [
            (
                0,
                negative_probability,
            ),
            (
                1,
                positive_probability,
            ),
        ]:
            rows.append(
                {
                    "recording_id":
                        f"{participant}::recording-{label}",
                    "participant":
                        participant,
                    "dataset":
                        "synthetic",
                    "task":
                        "stim01",
                    "label":
                        label,
                    "probability":
                        probability,
                    "threshold":
                        threshold,
                    "predicted_label":
                        int(
                            probability
                            >= threshold
                        ),
                }
            )

    return pd.DataFrame(
        rows
    )


def run_synthetic_smoke() -> dict[str, Any]:
    """Exercise fold-aware metrics, bootstrap, and report writing."""

    predictions = synthetic_fold_aware_predictions()

    metrics = fold_aware_recording_metrics(
        predictions
    )

    if metrics[
        "balanced_accuracy"
    ] != 1.0:
        raise RuntimeError(
            "Fold-aware synthetic balanced accuracy is not perfect."
        )

    global_predictions = (
        predictions[
            "probability"
        ].to_numpy(
            dtype=float
        )
        >= 0.5
    ).astype(int)

    global_balanced_accuracy = float(
        balanced_accuracy_score(
            predictions[
                "label"
            ].astype(int),
            global_predictions,
        )
    )

    if global_balanced_accuracy >= 1.0:
        raise RuntimeError(
            "Synthetic example does not distinguish global and fold thresholds."
        )

    first = fold_aware_participant_cluster_bootstrap(
        predictions,
        repetitions=
            50,
        random_seed=
            3407,
    )

    second = fold_aware_participant_cluster_bootstrap(
        predictions,
        repetitions=
            50,
        random_seed=
            3407,
    )

    pd.testing.assert_frame_equal(
        first,
        second,
    )

    with tempfile.TemporaryDirectory() as temporary:
        summary = write_fold_aware_report(
            recording_predictions=
                predictions,
            output_directory=
                Path(
                    temporary
                )
                / "report",
            protocol_identity=
                "synthetic_fold_aware",
            bootstrap_repetitions=
                50,
            bootstrap_seed=
                3407,
            confidence_level=
                0.95,
        )

        if summary[
            "pooled_recording_metrics"
        ][
            "balanced_accuracy"
        ] != 1.0:
            raise RuntimeError(
                "Synthetic report changed fold-aware performance."
            )

    return {
        "audit_identity":
            "bbbd_fold_aware_reporting_synthetic_smoke",
        "data_used":
            "synthetic_only",
        "bbbd_performance_observed":
            False,
        "fold_aware_balanced_accuracy":
            metrics[
                "balanced_accuracy"
            ],
        "incorrect_global_threshold_balanced_accuracy":
            global_balanced_accuracy,
        "participant_bootstrap_deterministic":
            True,
        "probabilities_shifted":
            False,
        "report_output_verified":
            True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the synthetic smoke for fold-aware BBBD reporting."
        )
    )

    parser.add_argument(
        "--synthetic-smoke",
        action="store_true",
        required=True,
    )

    arguments = parser.parse_args()

    if not arguments.synthetic_smoke:
        raise RuntimeError(
            "Only synthetic smoke mode is available from this entry point."
        )

    result = run_synthetic_smoke()

    print(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()