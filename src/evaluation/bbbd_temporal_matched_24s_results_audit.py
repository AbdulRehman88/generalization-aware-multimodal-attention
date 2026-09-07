"""Independent numerical audit of matched BBBD twenty-four-second reports."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.evaluation.bbbd_temporal_matched_results_audit import audit_reports


EXPECTED_MANIFEST_IDENTITY = "bbbd_temporal_matched_24s_reports_v1"
EXPECTED_ROLE = "post-primary temporal sensitivity"


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else Path.cwd() / path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for block in iter(
            lambda: stream.read(4 * 1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def audit_twenty_four_second_reports(
    report_root: str | Path,
) -> tuple[dict[str, Any], pd.DataFrame]:
    resolved_root = resolve_project_path(report_root)

    manifest_path = resolved_root / "reports_manifest.json"
    registry_path = resolved_root / "reports_registry.csv"

    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)

    if not registry_path.is_file():
        raise FileNotFoundError(registry_path)

    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )

    expected_manifest_values = {
        "identity": EXPECTED_MANIFEST_IDENTITY,
        "duration_seconds": 24,
        "scientific_role": EXPECTED_ROLE,
        "primary_four_second_392_recording_result_replaced": False,
        "report_count": 28,
        "within_experiment_reports": 14,
        "cross_experiment_reports": 14,
        "bootstrap_unit": "participant",
        "bootstrap_repetitions_per_report": 5000,
        "bootstrap_seed": 3407,
        "confidence_level": 0.95,
        "performance_observed": True,
    }

    for key, expected in expected_manifest_values.items():
        observed = manifest.get(key)

        if observed != expected:
            raise RuntimeError(
                f"Unexpected report manifest value for {key}: "
                f"{observed!r} versus {expected!r}"
            )

    registry = pd.read_csv(
        registry_path,
        low_memory=False,
    )

    if len(registry) != 28:
        raise RuntimeError(
            f"Expected 28 registry rows, observed {len(registry)}."
        )

    required_registry_columns = {
        "analysis_type",
        "dataset_or_direction",
        "path",
        "participant_count",
        "recording_count",
        "split_count",
        "bootstrap_repetitions",
        "bootstrap_seed",
        "confidence_level",
        "summary_sha256",
        "bootstrap_sha256",
    }

    missing_registry_columns = (
        required_registry_columns - set(registry.columns)
    )

    if missing_registry_columns:
        raise RuntimeError(
            "Report registry lacks required columns: "
            f"{sorted(missing_registry_columns)}"
        )

    if registry[
        [
            "analysis_type",
            "dataset_or_direction",
            "path",
        ]
    ].duplicated().any():
        raise RuntimeError(
            "Report registry contains duplicate group identities."
        )

    report_files = [
        path
        for path in resolved_root.rglob("*")
        if path.is_file()
    ]

    csv_files = [
        path
        for path in report_files
        if path.suffix.lower() == ".csv"
    ]

    json_files = [
        path
        for path in report_files
        if path.suffix.lower() == ".json"
    ]

    if len(report_files) != 142:
        raise RuntimeError(
            f"Expected 142 report files, observed {len(report_files)}."
        )

    if len(csv_files) != 113:
        raise RuntimeError(
            f"Expected 113 report CSV files, observed {len(csv_files)}."
        )

    if len(json_files) != 29:
        raise RuntimeError(
            f"Expected 29 report JSON files, observed {len(json_files)}."
        )

    payload, metrics = audit_reports(resolved_root)

    if payload.get("status") != "passed":
        raise RuntimeError(
            "Underlying independent numerical audit did not pass."
        )

    if payload.get("numerical_mismatches") != 0:
        raise RuntimeError(
            "Underlying numerical audit found mismatches."
        )

    if len(metrics) != 28:
        raise RuntimeError(
            f"Expected 28 metric rows, observed {len(metrics)}."
        )

    metrics = metrics.copy()

    metrics["duration_seconds"] = 24
    metrics["scientific_role"] = (
        "post_primary_temporal_sensitivity"
    )
    metrics[
        "primary_four_second_392_recording_analysis_replaced"
    ] = False

    if not metrics[
        "all_numerical_checks_passed"
    ].astype(bool).all():
        raise RuntimeError(
            "One or more report groups failed numerical validation."
        )

    best_rows = (
        metrics.loc[
            metrics[
                "descriptive_rank_within_setting"
            ].astype(int)
            == 1
        ]
        .sort_values(
            [
                "analysis_type",
                "dataset_or_direction",
            ],
            kind="stable",
        )
    )

    if len(best_rows) != 4:
        raise RuntimeError(
            "Expected one descriptive best path for each setting."
        )

    payload.update(
        {
            "audit_identity":
                "bbbd_temporal_matched_24s_metrics_v1",
            "source_commit":
                "2929c63",
            "status":
                "passed",
            "duration_seconds":
                24,
            "scientific_role":
                EXPECTED_ROLE,
            "primary_four_second_392_recording_analysis_replaced":
                False,
            "report_manifest_identity":
                EXPECTED_MANIFEST_IDENTITY,
            "report_groups_checked":
                28,
            "within_experiment_groups":
                14,
            "cross_experiment_groups":
                14,
            "bootstrap_repetitions_per_group":
                5000,
            "bootstrap_rows_checked":
                140000,
            "report_files_checked":
                142,
            "report_csv_files_checked":
                113,
            "report_json_files_checked":
                29,
            "numerical_mismatches":
                0,
            "performance_observed":
                True,
            "interpretation_boundary": (
                "BBBD attentive-versus-distracted "
                "experimental-condition classification; "
                "not isolated attention-state recognition."
            ),
            "temporal_interpretation_boundary": (
                "Twenty-four seconds is a prespecified post-primary "
                "temporal-sensitivity duration. It does not replace "
                "the primary four-second analysis and is not selected "
                "post hoc as a new operating point."
            ),
            "reports_manifest_sha256":
                sha256_file(manifest_path),
            "reports_registry_sha256":
                sha256_file(registry_path),
            "source_numerical_audit_module_sha256":
                sha256_file(
                    Path(
                        "src/evaluation/"
                        "bbbd_temporal_matched_results_audit.py"
                    )
                ),
            "best_path_by_setting": [
                {
                    "analysis_type":
                        str(row["analysis_type"]),
                    "dataset_or_direction":
                        str(row["dataset_or_direction"]),
                    "path":
                        str(row["path"]),
                    "pooled_balanced_accuracy":
                        float(row["pooled_balanced_accuracy"]),
                    "pooled_macro_f1":
                        float(row["pooled_macro_f1"]),
                    "pooled_roc_auc":
                        float(row["pooled_roc_auc"]),
                    "pooled_pr_auc":
                        float(row["pooled_pr_auc"]),
                }
                for _, row in best_rows.iterrows()
            ],
        }
    )

    return payload, metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Independently audit matched BBBD "
            "twenty-four-second reports."
        )
    )

    parser.add_argument(
        "--report-root",
        required=True,
    )

    parser.add_argument(
        "--output-json",
        required=True,
    )

    parser.add_argument(
        "--output-csv",
        required=True,
    )

    arguments = parser.parse_args()

    payload, metrics = audit_twenty_four_second_reports(
        arguments.report_root
    )

    output_json = resolve_project_path(
        arguments.output_json
    )

    output_csv = resolve_project_path(
        arguments.output_csv
    )

    output_json.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_json.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    metrics.to_csv(
        output_csv,
        index=False,
        float_format="%.15g",
    )

    print(
        "\n===== TWENTY-FOUR-SECOND NUMERICAL AUDIT SUMMARY ====="
    )

    print(
        "Report groups checked:",
        payload["report_groups_checked"],
    )

    print(
        "Bootstrap rows checked:",
        payload["bootstrap_rows_checked"],
    )

    print(
        "Numerical mismatches:",
        payload["numerical_mismatches"],
    )

    print(
        "\nDescriptive best path by setting:"
    )

    for record in payload["best_path_by_setting"]:
        print(
            f"{record['analysis_type']} | "
            f"{record['dataset_or_direction']} | "
            f"{record['path']} | "
            f"BA={record['pooled_balanced_accuracy']:.6f} | "
            f"F1={record['pooled_macro_f1']:.6f}"
        )

    print(
        "\nINDEPENDENT 24-SECOND NUMERICAL AUDIT PASSED"
    )


if __name__ == "__main__":
    main()
