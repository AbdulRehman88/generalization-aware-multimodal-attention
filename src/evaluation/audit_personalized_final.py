"""Integrity and participant-cluster uncertainty audit for the final protocol.

This audit does not fit, tune, select, or modify any model. It verifies the
already locked untouched-test artifacts and reports uncertainty using the
participant, rather than the temporally correlated window, as the resampling
unit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.core.config import load_revision_config
from src.evaluation.internal_grouped_pilot import (
    compute_metrics,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROBABILITY_COLUMNS = [
    "probability_class_0",
    "probability_class_1",
    "probability_class_2",
]

METRIC_NAMES = [
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "macro_precision",
    "macro_recall",
    "macro_roc_auc_ovr",
    "macro_pr_auc",
]


def resolve_project_path(
    value: str,
) -> Path:
    """Resolve a project-relative or absolute path."""

    path = Path(value)

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path.resolve()


def evaluate_prediction_frame(
    frame: pd.DataFrame,
) -> dict[str, float]:
    """Recompute metrics from stored labels and probabilities."""

    probabilities = frame[
        PROBABILITY_COLUMNS
    ].to_numpy(dtype=float)

    predictions = np.argmax(
        probabilities,
        axis=1,
    )

    return compute_metrics(
        frame[
            "true_label"
        ].to_numpy(dtype=int),
        predictions,
        probabilities,
    )


def validate_probabilities(
    frame: pd.DataFrame,
    name: str,
) -> None:
    """Verify finite, normalized multiclass probabilities."""

    probabilities = frame[
        PROBABILITY_COLUMNS
    ].to_numpy(dtype=float)

    if not np.isfinite(
        probabilities
    ).all():
        raise RuntimeError(
            f"{name}: probabilities contain non-finite values."
        )

    if (
        probabilities < -1e-12
    ).any():
        raise RuntimeError(
            f"{name}: probabilities contain negative values."
        )

    sums = probabilities.sum(
        axis=1
    )

    if not np.allclose(
        sums,
        np.ones(len(frame)),
        atol=1e-9,
        rtol=1e-9,
    ):
        maximum_error = float(
            np.max(
                np.abs(
                    sums - 1.0
                )
            )
        )

        raise RuntimeError(
            f"{name}: probability rows are not normalized; "
            f"maximum error={maximum_error}."
        )


def compare_metric_dictionary(
    recomputed: dict[str, float],
    stored: dict[str, Any],
    name: str,
) -> None:
    """Require exact numerical agreement with the locked summary."""

    for metric in METRIC_NAMES:
        expected = float(
            stored[metric]
        )

        observed = float(
            recomputed[metric]
        )

        if not np.isclose(
            observed,
            expected,
            atol=1e-12,
            rtol=1e-12,
        ):
            raise RuntimeError(
                f"{name}: metric mismatch for {metric}: "
                f"stored={expected}, recomputed={observed}."
            )


def participant_metrics(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Compute all metrics separately for every participant."""

    rows: list[dict[str, Any]] = []

    for participant, group in frame.groupby(
        "participant",
        sort=True,
    ):
        labels = set(
            group[
                "true_label"
            ].astype(int)
        )

        if labels != {
            0,
            1,
            2,
        }:
            raise RuntimeError(
                f"{participant}: participant test data "
                f"does not contain all three classes."
            )

        metrics = evaluate_prediction_frame(
            group
        )

        rows.append(
            {
                "participant":
                    str(participant),
                "decision_count":
                    int(len(group)),
                **metrics,
            }
        )

    return pd.DataFrame(rows)


def participant_cluster_bootstrap(
    frame: pd.DataFrame,
    *,
    iterations: int,
    random_seed: int,
) -> pd.DataFrame:
    """Bootstrap complete participant clusters with replacement."""

    if iterations <= 0:
        raise ValueError(
            "Bootstrap iterations must be positive."
        )

    participants = np.asarray(
        sorted(
            frame[
                "participant"
            ].astype(str).unique()
        ),
        dtype=object,
    )

    if len(participants) < 2:
        raise RuntimeError(
            "At least two participants are required."
        )

    grouped = {
        participant: frame.loc[
            frame[
                "participant"
            ].astype(str)
            == participant
        ].copy()
        for participant in participants
    }

    rng = np.random.default_rng(
        random_seed
    )

    rows: list[dict[str, float]] = []

    for iteration in range(
        iterations
    ):
        sampled = rng.choice(
            participants,
            size=len(participants),
            replace=True,
        )

        bootstrap_frame = pd.concat(
            [
                grouped[
                    str(participant)
                ]
                for participant in sampled
            ],
            ignore_index=True,
        )

        metrics = evaluate_prediction_frame(
            bootstrap_frame
        )

        rows.append(
            {
                "iteration":
                    int(iteration + 1),
                **metrics,
            }
        )

    return pd.DataFrame(rows)


def summarize_bootstrap(
    bootstrap: pd.DataFrame,
) -> dict[str, dict[str, float]]:
    """Return percentile confidence intervals for all metrics."""

    summary: dict[
        str,
        dict[str, float],
    ] = {}

    for metric in METRIC_NAMES:
        values = bootstrap[
            metric
        ].to_numpy(dtype=float)

        summary[metric] = {
            "bootstrap_mean":
                float(
                    np.mean(values)
                ),
            "bootstrap_standard_error":
                float(
                    np.std(
                        values,
                        ddof=1,
                    )
                ),
            "ci_95_lower":
                float(
                    np.percentile(
                        values,
                        2.5,
                    )
                ),
            "ci_95_upper":
                float(
                    np.percentile(
                        values,
                        97.5,
                    )
                ),
        }

    return summary


def validate_selected_features(
    selected_features: pd.DataFrame,
) -> list[str]:
    """Verify that no metadata or protocol variable enters the model."""

    if list(
        selected_features.columns
    ) != ["feature"]:
        raise RuntimeError(
            "Selected-feature file has an unexpected schema."
        )

    features = selected_features[
        "feature"
    ].astype(str).tolist()

    if not features:
        raise RuntimeError(
            "Selected-feature list is empty."
        )

    if len(features) != len(
        set(features)
    ):
        raise RuntimeError(
            "Selected-feature list contains duplicates."
        )

    forbidden_tokens = [
        "__participant",
        "__phase",
        "__label",
        "__window_index",
        "__segment_id",
        "__role",
    ]

    invalid = [
        feature
        for feature in features
        if (
            not feature.startswith(
                (
                    "context_mean__",
                    "context_std__",
                    "context_slope__",
                )
            )
            or any(
                token in feature
                for token in forbidden_tokens
            )
        )
    ]

    if invalid:
        raise RuntimeError(
            "Protocol or metadata variables entered the feature set: "
            f"{invalid[:10]}"
        )

    return features


def validate_role_counts(
    role_counts: pd.DataFrame,
) -> dict[str, Any]:
    """Verify the expected chronology for all participant-phase groups."""

    required = {
        "participant",
        "phase",
        "role",
        "rows",
    }

    missing = required - set(
        role_counts.columns
    )

    if missing:
        raise RuntimeError(
            f"Role-count file lacks columns: {sorted(missing)}"
        )

    expected_by_phase = {
        1: {
            "calibration": 11,
            "guard_1": 5,
            "validation": 15,
            "guard_2": 5,
            "test": 35,
        },
        2: {
            "calibration": 11,
            "guard_1": 5,
            "validation": 15,
            "guard_2": 5,
            "test": 65,
        },
        3: {
            "calibration": 11,
            "guard_1": 5,
            "validation": 15,
            "guard_2": 5,
            "test": 35,
        },
    }

    for row in role_counts.itertuples(
        index=False
    ):
        phase = int(row.phase)
        role = str(row.role)
        observed = int(row.rows)

        if phase not in expected_by_phase:
            raise RuntimeError(
                f"Unexpected phase in role audit: {phase}"
            )

        expected = expected_by_phase[
            phase
        ].get(role)

        if expected is None:
            raise RuntimeError(
                f"Unexpected role in phase {phase}: {role}"
            )

        if observed != expected:
            raise RuntimeError(
                f"{row.participant}, phase {phase}, role {role}: "
                f"expected {expected}, observed {observed}."
            )

    expected_rows = (
        13
        * 3
        * 5
    )

    if len(role_counts) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} participant-phase-role "
            f"rows, observed {len(role_counts)}."
        )

    return {
        "participant_phase_role_rows":
            int(len(role_counts)),
        "chronology_matches_expected_design":
            True,
        "expected_test_windows_per_participant":
            135,
    }


def run_audit(
    config: dict[str, Any],
    *,
    write_outputs: bool,
) -> dict[str, Any]:
    """Audit the locked personalized-final results."""

    protocol = config[
        "training"
    ][
        "personalized_final"
    ]

    result_directory = (
        resolve_project_path(
            protocol[
                "output_dir"
            ]
        )
    )

    audit_directory = (
        resolve_project_path(
            protocol[
                "audit_output_dir"
            ]
        )
    )

    bootstrap_iterations = int(
        protocol[
            "audit_bootstrap_iterations"
        ]
    )

    summary_path = (
        result_directory
        / "final_summary.json"
    )

    prediction_path = (
        result_directory
        / "untouched_test_predictions.csv"
    )

    phase_path = (
        result_directory
        / "phase_level_predictions.csv"
    )

    selected_path = (
        result_directory
        / "final_selected_features.csv"
    )

    role_count_path = (
        result_directory
        / "participant_phase_role_counts.csv"
    )

    required_paths = [
        summary_path,
        prediction_path,
        phase_path,
        selected_path,
        role_count_path,
    ]

    missing_paths = [
        path
        for path in required_paths
        if not path.is_file()
    ]

    if missing_paths:
        raise FileNotFoundError(
            "Missing final-result artifacts: "
            f"{missing_paths}"
        )

    summary = json.loads(
        summary_path.read_text(
            encoding="utf-8"
        )
    )

    predictions = pd.read_csv(
        prediction_path,
        low_memory=False,
    )

    phase_predictions = pd.read_csv(
        phase_path,
        low_memory=False,
    )

    selected_features = pd.read_csv(
        selected_path,
        low_memory=False,
    )

    role_counts = pd.read_csv(
        role_count_path,
        low_memory=False,
    )

    if predictions[
        "segment_id"
    ].duplicated().any():
        raise RuntimeError(
            "Untouched-test predictions contain duplicate segments."
        )

    if len(predictions) != int(
        summary[
            "untouched_test_decisions"
        ]
    ):
        raise RuntimeError(
            "Untouched-test row count does not match the summary."
        )

    if predictions[
        "participant"
    ].nunique() != int(
        summary[
            "participant_count"
        ]
    ):
        raise RuntimeError(
            "Participant count does not match the summary."
        )

    validate_probabilities(
        predictions,
        "Untouched test",
    )

    validate_probabilities(
        phase_predictions,
        "Phase level",
    )

    recomputed_test_metrics = (
        evaluate_prediction_frame(
            predictions
        )
    )

    recomputed_phase_metrics = (
        evaluate_prediction_frame(
            phase_predictions
        )
    )

    compare_metric_dictionary(
        recomputed_test_metrics,
        summary[
            "untouched_test_metrics"
        ],
        "Untouched test",
    )

    compare_metric_dictionary(
        recomputed_phase_metrics,
        summary[
            "phase_level_metrics"
        ],
        "Phase level",
    )

    features = validate_selected_features(
        selected_features
    )

    chronology_audit = validate_role_counts(
        role_counts
    )

    per_participant = participant_metrics(
        predictions
    )

    bootstrap = participant_cluster_bootstrap(
        predictions,
        iterations=bootstrap_iterations,
        random_seed=20260728,
    )

    bootstrap_summary = summarize_bootstrap(
        bootstrap
    )

    phase_bootstrap = participant_cluster_bootstrap(
        phase_predictions,
        iterations=bootstrap_iterations,
        random_seed=20260729,
    )

    phase_bootstrap_summary = (
        summarize_bootstrap(
            phase_bootstrap
        )
    )

    smoothing_windows = int(
        summary[
            "validation_configuration"
        ][
            "smoothing_windows"
        ]
    )

    smoothing_seconds = int(
        summary[
            "validation_configuration"
        ][
            "probability_history_seconds"
        ]
    )

    adjacent_history_overlap = (
        smoothing_windows - 1
    ) / smoothing_windows

    participant_metric_summary = {
        metric: {
            "participant_mean":
                float(
                    per_participant[
                        metric
                    ].mean()
                ),
            "participant_standard_deviation":
                float(
                    per_participant[
                        metric
                    ].std(
                        ddof=1
                    )
                ),
            "participant_minimum":
                float(
                    per_participant[
                        metric
                    ].min()
                ),
            "participant_maximum":
                float(
                    per_participant[
                        metric
                    ].max()
                ),
        }
        for metric in METRIC_NAMES
    }

    audit_summary = {
        "audit_status":
            "passed",
        "model_or_parameter_refitting_performed":
            False,
        "result_directory":
            str(result_directory),
        "participant_count":
            int(
                predictions[
                    "participant"
                ].nunique()
            ),
        "untouched_test_rows":
            int(len(predictions)),
        "unique_untouched_test_segments":
            int(
                predictions[
                    "segment_id"
                ].nunique()
            ),
        "phase_level_decisions":
            int(
                len(
                    phase_predictions
                )
            ),
        "selected_feature_count":
            int(len(features)),
        "metadata_feature_leakage_detected":
            False,
        "probability_validation_passed":
            True,
        "stored_metric_recalculation_passed":
            True,
        "chronology_audit":
            chronology_audit,
        "smoothing_windows":
            smoothing_windows,
        "smoothing_seconds":
            smoothing_seconds,
        "adjacent_decision_history_overlap_fraction":
            float(
                adjacent_history_overlap
            ),
        "independence_warning": (
            "The 1,755 four-second-cadence outputs are temporally "
            "correlated because each selected prediction averages "
            "up to 120 seconds of causal probability history. "
            "Participant-cluster confidence intervals, rather than "
            "row-level intervals, should be reported."
        ),
        "deployment_scope": (
            "Within-cohort, labeled, participant-specific adaptation "
            "using calibration and validation samples from all three "
            "classes. This is not calibration-free unseen-participant "
            "generalization."
        ),
        "recomputed_untouched_test_metrics":
            recomputed_test_metrics,
        "recomputed_phase_level_metrics":
            recomputed_phase_metrics,
        "participant_metric_summary":
            participant_metric_summary,
        "participant_cluster_bootstrap_95_ci":
            bootstrap_summary,
        "phase_participant_cluster_bootstrap_95_ci":
            phase_bootstrap_summary,
    }

    report_lines = [
        "# Personalized Final Protocol Integrity Audit",
        "",
        "## Audit verdict",
        "",
        "All stored untouched-test and phase-level metrics were "
        "recomputed successfully from the saved predictions. No "
        "duplicate test segments, invalid probabilities, metadata "
        "features, or chronology violations were detected.",
        "",
        "## Locked untouched-test performance",
        "",
        (
            f"- Accuracy: "
            f"{recomputed_test_metrics['accuracy']:.4f}"
        ),
        (
            f"- Balanced accuracy: "
            f"{recomputed_test_metrics['balanced_accuracy']:.4f}"
        ),
        (
            f"- Macro-F1: "
            f"{recomputed_test_metrics['macro_f1']:.4f}"
        ),
        (
            f"- Macro ROC-AUC: "
            f"{recomputed_test_metrics['macro_roc_auc_ovr']:.4f}"
        ),
        (
            f"- Macro PR-AUC: "
            f"{recomputed_test_metrics['macro_pr_auc']:.4f}"
        ),
        "",
        "## Participant-cluster 95% confidence intervals",
        "",
    ]

    for metric in (
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "macro_roc_auc_ovr",
        "macro_pr_auc",
    ):
        interval = bootstrap_summary[
            metric
        ]

        report_lines.append(
            f"- {metric}: "
            f"{interval['ci_95_lower']:.4f} to "
            f"{interval['ci_95_upper']:.4f}"
        )

    report_lines.extend(
        [
            "",
            "## Interpretation constraints",
            "",
            (
                f"- Predictions occur every four seconds but use "
                f"{smoothing_seconds} seconds of causal probability "
                f"history."
            ),
            (
                f"- Adjacent predictions share approximately "
                f"{adjacent_history_overlap * 100:.1f}% of their "
                f"probability-history windows."
            ),
            "- The 1,755 outputs must not be treated as 1,755 "
            "independent observations.",
            "- The valid inferential unit is the participant; "
            "participant-cluster uncertainty is therefore reported.",
            "- The protocol requires labeled calibration and validation "
            "samples from low, mid, and high states for each participant.",
            "- The result must not be described as calibration-free, "
            "cross-participant, LOSO, or four-second independent "
            "classification performance.",
            "",
        ]
    )

    if write_outputs:
        temporary_directory = (
            audit_directory.with_name(
                audit_directory.name
                + ".building"
            )
        )

        if audit_directory.exists():
            raise FileExistsError(
                "Refusing to overwrite existing audit output: "
                f"{audit_directory}"
            )

        if temporary_directory.exists():
            raise FileExistsError(
                "Temporary audit directory exists: "
                f"{temporary_directory}"
            )

        temporary_directory.mkdir(
            parents=True,
            exist_ok=False,
        )

        per_participant.to_csv(
            temporary_directory
            / "per_participant_metrics.csv",
            index=False,
        )

        pd.DataFrame(
            [
                {
                    "metric": metric,
                    **values,
                }
                for metric, values
                in bootstrap_summary.items()
            ]
        ).to_csv(
            temporary_directory
            / "participant_cluster_bootstrap_ci.csv",
            index=False,
        )

        pd.DataFrame(
            [
                {
                    "metric": metric,
                    **values,
                }
                for metric, values
                in phase_bootstrap_summary.items()
            ]
        ).to_csv(
            temporary_directory
            / "phase_participant_cluster_bootstrap_ci.csv",
            index=False,
        )

        (
            temporary_directory
            / "audit_summary.json"
        ).write_text(
            json.dumps(
                audit_summary,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        (
            temporary_directory
            / "audit_report.md"
        ).write_text(
            "\n".join(
                report_lines
            ),
            encoding="utf-8",
        )

        temporary_directory.replace(
            audit_directory
        )

    print(
        "\n===== FINAL RESULT INTEGRITY AUDIT ====="
    )

    print(
        "Audit status:",
        audit_summary[
            "audit_status"
        ],
    )

    print(
        "Stored metrics recomputed:",
        audit_summary[
            "stored_metric_recalculation_passed"
        ],
    )

    print(
        "Metadata leakage detected:",
        audit_summary[
            "metadata_feature_leakage_detected"
        ],
    )

    print(
        "Unique test segments:",
        audit_summary[
            "unique_untouched_test_segments"
        ],
    )

    print(
        "Selected features:",
        audit_summary[
            "selected_feature_count"
        ],
    )

    print(
        "\nParticipant-level metric range:"
    )

    print(
        per_participant[
            [
                "participant",
                "decision_count",
                "accuracy",
                "balanced_accuracy",
                "macro_f1",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\nParticipant-cluster 95% confidence intervals:"
    )

    for metric in (
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "macro_roc_auc_ovr",
        "macro_pr_auc",
    ):
        interval = bootstrap_summary[
            metric
        ]

        print(
            f"{metric:24s} "
            f"{interval['ci_95_lower']:.4f} "
            f"to "
            f"{interval['ci_95_upper']:.4f}"
        )

    print(
        "\nAdjacent decision history overlap:",
        f"{adjacent_history_overlap * 100:.1f}%",
    )

    print(
        "Audit output:",
        audit_directory,
    )

    return audit_summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit the locked personalized-final result."
        )
    )

    parser.add_argument(
        "--write",
        action="store_true",
        help="Write audit artifacts.",
    )

    arguments = parser.parse_args()

    config = load_revision_config(
        check_input_paths=True
    )

    run_audit(
        config,
        write_outputs=arguments.write,
    )


if __name__ == "__main__":
    main()