"""Final integrity audit for locked internal attention-detection results."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.core.config import load_revision_config
from src.evaluation.internal_grouped_pilot import compute_metrics


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROBABILITY_COLUMNS = [
    "probability_class_0",
    "probability_class_1",
    "probability_class_2",
]

PRIMARY_METRICS = [
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "macro_roc_auc_ovr",
    "macro_pr_auc",
]


def resolve_project_path(value: str) -> Path:
    path = Path(value)

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path.resolve()


def validate_probabilities(probabilities: np.ndarray) -> None:
    values = np.asarray(probabilities, dtype=float)

    if values.ndim != 2 or values.shape[1] != 3:
        raise RuntimeError("Expected a three-class probability matrix.")

    if not np.isfinite(values).all():
        raise RuntimeError("Probabilities contain nonfinite values.")

    if (values < -1e-12).any():
        raise RuntimeError("Probabilities contain negative values.")

    if not np.allclose(
        values.sum(axis=1),
        np.ones(len(values)),
        atol=1e-8,
    ):
        raise RuntimeError("Probability rows do not sum to one.")


def recompute_metrics(frame: pd.DataFrame) -> dict[str, float]:
    probabilities = frame[PROBABILITY_COLUMNS].to_numpy(dtype=float)
    validate_probabilities(probabilities)

    labels = frame["true_label"].to_numpy(dtype=int)
    predictions = np.argmax(probabilities, axis=1)

    if "predicted_label" in frame.columns:
        recorded = frame["predicted_label"].to_numpy(dtype=int)

        if not np.array_equal(predictions, recorded):
            raise RuntimeError(
                "Recorded predictions differ from probability argmax."
            )

    return compute_metrics(labels, predictions, probabilities)


def validate_outer_structure(
    frame: pd.DataFrame,
    protocol_name: str,
) -> None:
    required = {
        "segment_id",
        "participant",
        "outer_fold",
        "true_label",
        "predicted_label",
        *PROBABILITY_COLUMNS,
    }

    missing = required - set(frame.columns)

    if missing:
        raise RuntimeError(
            f"{protocol_name} lacks columns: {sorted(missing)}"
        )

    if frame["segment_id"].duplicated().any():
        raise RuntimeError(
            f"{protocol_name} contains duplicate test segments."
        )

    if frame["participant"].astype(str).nunique() != 13:
        raise RuntimeError(
            f"{protocol_name} does not include all 13 participants."
        )

    if "P10" not in set(frame["participant"].astype(str)):
        raise RuntimeError(f"{protocol_name} excludes P10.")

    fold_participant_counts = (
        frame.groupby("outer_fold")["participant"]
        .nunique()
    )

    if not (fold_participant_counts == 1).all():
        raise RuntimeError(
            f"{protocol_name} has a fold containing multiple participants."
        )

    participant_fold_counts = (
        frame.groupby("participant")["outer_fold"]
        .nunique()
    )

    if not (participant_fold_counts == 1).all():
        raise RuntimeError(
            f"{protocol_name} evaluates a participant in multiple folds."
        )


def compare_metrics(
    observed: dict[str, float],
    expected: dict[str, float],
    protocol_name: str,
    tolerance: float = 1e-12,
) -> None:
    """Compare locked metrics with serialization-aware tolerances.

    Label-derived metrics must reproduce essentially exactly. ROC-AUC and
    PR-AUC are computed from probabilities reloaded from CSV and may differ
    at approximately the sixth decimal place because of text serialization.
    """

    probability_metric_tolerance = 1e-5

    for metric in PRIMARY_METRICS:
        observed_value = float(observed[metric])
        expected_value = float(expected[metric])

        metric_tolerance = (
            probability_metric_tolerance
            if metric in {
                "macro_roc_auc_ovr",
                "macro_pr_auc",
            }
            else tolerance
        )

        absolute_difference = abs(
            observed_value - expected_value
        )

        if absolute_difference > metric_tolerance:
            raise RuntimeError(
                f"{protocol_name} {metric} changed: "
                f"expected {expected_value}, "
                f"observed {observed_value}, "
                f"difference {absolute_difference}, "
                f"tolerance {metric_tolerance}"
            )


def cluster_bootstrap(
    frame: pd.DataFrame,
    repetitions: int,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    participants = sorted(
        frame["participant"].astype(str).unique()
    )

    participant_indices = {
        participant: frame.index[
            frame["participant"].astype(str) == participant
        ].to_numpy(dtype=int)
        for participant in participants
    }

    participant_metric_rows: list[dict[str, Any]] = []

    for participant in participants:
        participant_frame = frame.loc[
            participant_indices[participant]
        ]

        metrics = recompute_metrics(participant_frame)

        participant_metric_rows.append(
            {
                "participant": participant,
                **{
                    metric: float(metrics[metric])
                    for metric in PRIMARY_METRICS
                },
            }
        )

    participant_metrics = pd.DataFrame(participant_metric_rows)

    random_generator = np.random.default_rng(random_seed)
    bootstrap_rows: list[dict[str, Any]] = []

    for repetition in range(repetitions):
        sampled_participants = random_generator.choice(
            participants,
            size=len(participants),
            replace=True,
        )

        sampled_indices = np.concatenate(
            [
                participant_indices[str(participant)]
                for participant in sampled_participants
            ]
        )

        sampled_frame = frame.loc[sampled_indices]
        pooled_metrics = recompute_metrics(sampled_frame)

        sampled_participant_metrics = participant_metrics.set_index(
            "participant"
        ).loc[
            [str(participant) for participant in sampled_participants]
        ]

        row: dict[str, Any] = {"repetition": repetition}

        for metric in PRIMARY_METRICS:
            row[f"pooled_{metric}"] = float(
                pooled_metrics[metric]
            )

            row[f"participant_mean_{metric}"] = float(
                sampled_participant_metrics[metric].mean()
            )

        bootstrap_rows.append(row)

    return participant_metrics, pd.DataFrame(bootstrap_rows)


def confidence_interval_table(
    protocol_name: str,
    bootstrap: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for aggregation in ["pooled", "participant_mean"]:
        for metric in PRIMARY_METRICS:
            column = f"{aggregation}_{metric}"
            values = bootstrap[column].to_numpy(dtype=float)

            rows.append(
                {
                    "protocol": protocol_name,
                    "aggregation": aggregation,
                    "metric": metric,
                    "bootstrap_mean": float(np.mean(values)),
                    "ci_2_5": float(np.percentile(values, 2.5)),
                    "ci_97_5": float(np.percentile(values, 97.5)),
                }
            )

    return pd.DataFrame(rows)


def verify_sha256_manifest(lock_directory: Path) -> pd.DataFrame:
    manifest_path = lock_directory / "sha256_manifest.csv"

    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"SHA256 manifest not found: {manifest_path}"
        )

    manifest = pd.read_csv(manifest_path, low_memory=False)
    rows: list[dict[str, Any]] = []

    for row in manifest.itertuples(index=False):
        relative_path = str(row.RelativePath)
        expected_hash = str(row.SHA256).upper()
        file_path = PROJECT_ROOT / relative_path

        if not file_path.is_file():
            raise FileNotFoundError(
                f"Locked result file is missing: {relative_path}"
            )

        digest = hashlib.sha256(file_path.read_bytes()).hexdigest().upper()
        matches = digest == expected_hash

        rows.append(
            {
                "relative_path": relative_path,
                "expected_sha256": expected_hash,
                "observed_sha256": digest,
                "matches": matches,
            }
        )

        if not matches:
            raise RuntimeError(
                f"SHA256 mismatch for {relative_path}"
            )

    return pd.DataFrame(rows)


def run_final_audit(
    config: dict[str, Any],
    *,
    write_outputs: bool,
) -> dict[str, Any]:
    audit_config = config["training"]["locked_internal_results_audit"]

    repetitions = int(audit_config["bootstrap_repetitions"])
    random_seed = int(audit_config["bootstrap_random_seed"])
    few_shot_budget = int(
        audit_config["few_shot_budget_seconds"]
    )

    lock_directory = resolve_project_path(
        audit_config["lock_directory"]
    )

    output_directory = resolve_project_path(
        audit_config["output_dir"]
    )

    strict_summary = json.loads(
        (
            lock_directory / "strict_loso_summary.json"
        ).read_text(encoding="utf-8")
    )

    baseline_summary = json.loads(
        (
            lock_directory / "baseline_calibrated_loso_summary.json"
        ).read_text(encoding="utf-8")
    )

    few_shot_summary = json.loads(
        (
            lock_directory / "few_shot_subject_adaptive_summary.json"
        ).read_text(encoding="utf-8")
    )

    p10_summary = json.loads(
        (
            lock_directory / "p10_audit_summary.json"
        ).read_text(encoding="utf-8")
    )

    protocol_definitions = {
        "strict_calibration_free_loso": {
            "prediction_path": resolve_project_path(
                "outputs/revision/evaluation/"
                "internal_xr_loso_long_window/"
                "outer_loso_predictions.csv"
            ),
            "expected_metrics": strict_summary["pooled_metrics"],
        },
        "baseline_calibrated_loso": {
            "prediction_path": resolve_project_path(
                "outputs/revision/evaluation/"
                "internal_xr_baseline_calibrated_loso/"
                "outer_predictions.csv"
            ),
            "expected_metrics": baseline_summary["pooled_metrics"],
        },
        "few_shot_120_seconds_per_class": {
            "prediction_path": resolve_project_path(
                "outputs/revision/evaluation/"
                "internal_xr_few_shot_subject_adaptive_loso/"
                "outer_predictions.csv"
            ),
            "expected_metrics": few_shot_summary[
                "budgets"
            ][str(few_shot_budget)]["pooled_metrics"],
        },
    }

    recomputation_rows: list[dict[str, Any]] = []
    participant_tables: list[pd.DataFrame] = []
    confidence_tables: list[pd.DataFrame] = []
    protocol_summaries: dict[str, Any] = {}

    for protocol_index, (
        protocol_name,
        definition,
    ) in enumerate(protocol_definitions.items()):
        prediction_path = definition["prediction_path"]

        if not prediction_path.is_file():
            raise FileNotFoundError(
                f"Prediction file not found: {prediction_path}"
            )

        frame = pd.read_csv(prediction_path, low_memory=False)

        if protocol_name.startswith("few_shot"):
            frame = frame.loc[
                frame["calibration_budget_seconds"].astype(int)
                == few_shot_budget
            ].copy()

        frame = frame.reset_index(drop=True)

        validate_outer_structure(frame, protocol_name)
        observed_metrics = recompute_metrics(frame)

        compare_metrics(
            observed_metrics,
            definition["expected_metrics"],
            protocol_name,
        )

        participant_metrics, bootstrap = cluster_bootstrap(
            frame,
            repetitions,
            random_seed + protocol_index,
        )

        participant_metrics.insert(
            0,
            "protocol",
            protocol_name,
        )

        participant_tables.append(participant_metrics)

        confidence_tables.append(
            confidence_interval_table(
                protocol_name,
                bootstrap,
            )
        )

        for metric in PRIMARY_METRICS:
            recomputation_rows.append(
                {
                    "protocol": protocol_name,
                    "metric": metric,
                    "recorded_value": float(
                        definition["expected_metrics"][metric]
                    ),
                    "recomputed_value": float(
                        observed_metrics[metric]
                    ),
                    "absolute_difference": abs(
                        float(definition["expected_metrics"][metric])
                        - float(observed_metrics[metric])
                    ),
                }
            )

        protocol_summaries[protocol_name] = {
            "participants": int(
                frame["participant"].astype(str).nunique()
            ),
            "test_decisions": int(len(frame)),
            "p10_included": True,
            "recomputed_metrics": {
                metric: float(observed_metrics[metric])
                for metric in PRIMARY_METRICS
            },
        }

    split_audit_path = resolve_project_path(
        "outputs/revision/evaluation/"
        "internal_xr_few_shot_subject_adaptive_loso/"
        "calibration_test_split_audit.csv"
    )

    split_audit = pd.read_csv(split_audit_path, low_memory=False)

    outer_split_audit = split_audit.loc[
        (
            split_audit["calibration_budget_seconds"].astype(int)
            == few_shot_budget
        )
        & (
            split_audit["scope"].astype(str)
            == "outer_test"
        )
    ].copy()

    if len(outer_split_audit) != 39:
        raise RuntimeError(
            "Expected 39 outer few-shot phase split records."
        )

    if not (
        outer_split_audit["first_test_start_seconds"].astype(float)
        >= (
            outer_split_audit[
                "calibration_raw_end_seconds"
            ].astype(float)
            + outer_split_audit["guard_seconds"].astype(float)
        )
    ).all():
        raise RuntimeError(
            "Few-shot calibration and test intervals are not guard-separated."
        )

    if p10_summary["objective_exclusion_supported"]:
        raise RuntimeError(
            "Locked P10 audit unexpectedly supports exclusion."
        )

    hash_audit = verify_sha256_manifest(lock_directory)

    recomputation_table = pd.DataFrame(recomputation_rows)
    participant_table = pd.concat(
        participant_tables,
        ignore_index=True,
    )

    confidence_table = pd.concat(
        confidence_tables,
        ignore_index=True,
    )

    summary = {
        "audit_type": "locked_internal_results_integrity_audit",
        "audit_passed": True,
        "bootstrap_repetitions": repetitions,
        "bootstrap_unit": "participant",
        "participant_count": 13,
        "p10_included_in_all_protocols": True,
        "p10_objective_exclusion_supported": False,
        "sha256_files_verified": int(len(hash_audit)),
        "few_shot_calibration_budget_seconds_per_class":
            few_shot_budget,
        "few_shot_outer_phase_splits_verified": int(
            len(outer_split_audit)
        ),
        "protocols": protocol_summaries,
        "interpretation": (
            "Confidence intervals resample participants as clusters. "
            "Overlapping windows are retained together within each sampled "
            "participant and are not treated as independent observations."
        ),
    }

    if write_outputs:
        if output_directory.exists():
            raise FileExistsError(
                f"Refusing to overwrite audit output: {output_directory}"
            )

        temporary = output_directory.with_name(
            output_directory.name + ".building"
        )

        if temporary.exists():
            raise FileExistsError(
                f"Temporary audit directory exists: {temporary}"
            )

        temporary.mkdir(parents=True, exist_ok=False)

        recomputation_table.to_csv(
            temporary / "metric_recomputation.csv",
            index=False,
        )

        confidence_table.to_csv(
            temporary / "participant_cluster_bootstrap_ci.csv",
            index=False,
        )

        participant_table.to_csv(
            temporary / "participant_metrics.csv",
            index=False,
        )

        hash_audit.to_csv(
            temporary / "sha256_verification.csv",
            index=False,
        )

        outer_split_audit.to_csv(
            temporary / "few_shot_120s_split_verification.csv",
            index=False,
        )

        (
            temporary / "integrity_audit_summary.json"
        ).write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        temporary.replace(output_directory)

    print("\n===== FINAL INTERNAL INTEGRITY AUDIT =====")
    print("Audit passed:", summary["audit_passed"])
    print("SHA256 files verified:", len(hash_audit))
    print("Participants verified:", 13)
    print("P10 retained:", True)
    print("Few-shot split records verified:", len(outer_split_audit))

    print("\n===== METRIC RECOMPUTATION =====")
    print(recomputation_table.to_string(index=False))

    print("\n===== PARTICIPANT-CLUSTER 95% CONFIDENCE INTERVALS =====")
    print(
        confidence_table.loc[
            confidence_table["aggregation"] == "pooled"
        ].to_string(index=False)
    )

    if write_outputs:
        print("\nOutput directory:", output_directory)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit locked internal evaluation results."
    )

    parser.add_argument(
        "--write",
        action="store_true",
        help="Write complete audit artifacts.",
    )

    arguments = parser.parse_args()

    config = load_revision_config(check_input_paths=True)

    run_final_audit(config, write_outputs=arguments.write)


if __name__ == "__main__":
    main()