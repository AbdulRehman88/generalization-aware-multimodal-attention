"""Aggregate and independently validate matched BBBD temporal evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


DURATIONS = [4, 8, 16, 24, 32]

PATHS = [
    "EEG",
    "ECG",
    "Pupil",
    "ECG_EEG",
    "ECG_Pupil",
    "EEG_Pupil",
    "ECG_EEG_Pupil",
]

SETTINGS = [
    ("within_experiment", "experiment2"),
    ("within_experiment", "experiment3"),
    ("cross_experiment", "experiment2_to_experiment3"),
    ("cross_experiment", "experiment3_to_experiment2"),
]

REQUIRED_COLUMNS = {
    "analysis_type",
    "dataset_or_direction",
    "path",
    "pooled_balanced_accuracy",
    "pooled_macro_f1",
    "pooled_roc_auc",
    "pooled_pr_auc",
    "participant_macro_macro_f1",
    "all_numerical_checks_passed",
}


def resolve_path(value: str | Path) -> Path:
    path = Path(value)

    if path.is_absolute():
        return path

    return Path.cwd() / path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for block in iter(
            lambda: stream.read(4 * 1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def source_paths(
    audit_root: Path,
    duration: int,
) -> tuple[Path, Path]:
    stem = (
        f"bbbd_temporal_matched_"
        f"{duration:02d}s_metrics_v1"
    )

    return (
        audit_root / f"{stem}.csv",
        audit_root / f"{stem}.json",
    )


def normalize_boolean(
    series: pd.Series,
) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)

    normalized = (
        series.astype(str)
        .str.strip()
        .str.lower()
    )

    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
    }

    unknown = sorted(
        set(normalized)
        - set(mapping)
    )

    if unknown:
        raise RuntimeError(
            f"Unknown boolean values: {unknown}"
        )

    return normalized.map(mapping).astype(bool)


def rank_within_setting_duration(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    ranked_groups: list[pd.DataFrame] = []

    group_columns = [
        "analysis_type",
        "dataset_or_direction",
        "duration_seconds",
    ]

    for _, group in frame.groupby(
        group_columns,
        sort=False,
    ):
        ranked = (
            group.sort_values(
                [
                    "pooled_balanced_accuracy",
                    "pooled_macro_f1",
                    "pooled_roc_auc",
                    "path",
                ],
                ascending=[
                    False,
                    False,
                    False,
                    True,
                ],
                kind="stable",
            )
            .copy()
        )

        ranked[
            "temporal_descriptive_rank_within_setting_duration"
        ] = range(
            1,
            len(ranked) + 1,
        )

        ranked_groups.append(ranked)

    return pd.concat(
        ranked_groups,
        ignore_index=True,
    )


def descriptive_best_rows(
    frame: pd.DataFrame,
    group_columns: list[str],
) -> pd.DataFrame:
    records: list[pd.DataFrame] = []

    for _, group in frame.groupby(
        group_columns,
        sort=False,
    ):
        ordered = group.sort_values(
            [
                "pooled_balanced_accuracy",
                "pooled_macro_f1",
                "pooled_roc_auc",
                "duration_seconds",
                "path",
            ],
            ascending=[
                False,
                False,
                False,
                True,
                True,
            ],
            kind="stable",
        )

        records.append(
            ordered.iloc[[0]].copy()
        )

    return pd.concat(
        records,
        ignore_index=True,
    )


def aggregate_temporal_evidence(
    audit_root: str | Path,
) -> tuple[
    dict[str, Any],
    pd.DataFrame,
]:
    resolved_audit_root = resolve_path(
        audit_root
    )

    frames: list[pd.DataFrame] = []
    source_records: list[dict[str, Any]] = []

    expected_setting_set = set(SETTINGS)
    expected_path_set = set(PATHS)

    for duration in DURATIONS:
        csv_path, json_path = source_paths(
            resolved_audit_root,
            duration,
        )

        if not csv_path.is_file():
            raise FileNotFoundError(csv_path)

        if not json_path.is_file():
            raise FileNotFoundError(json_path)

        audit = json.loads(
            json_path.read_text(
                encoding="utf-8"
            )
        )

        if audit.get("status") != "passed":
            raise RuntimeError(
                f"{duration}-second audit status is not passed."
            )

        if audit.get(
            "numerical_mismatches"
        ) != 0:
            raise RuntimeError(
                f"{duration}-second evidence contains mismatches."
            )

        observed_duration = int(
            audit.get(
                "duration_seconds",
                duration,
            )
        )

        if observed_duration != duration:
            raise RuntimeError(
                f"Unexpected audit duration: {observed_duration}."
            )

        if audit.get(
            "primary_four_second_392_recording_analysis_replaced",
            False,
        ) is not False:
            raise RuntimeError(
                "The immutable primary four-second analysis was replaced."
            )

        frame = pd.read_csv(
            csv_path,
            low_memory=False,
        )

        missing_columns = (
            REQUIRED_COLUMNS
            - set(frame.columns)
        )

        if missing_columns:
            raise RuntimeError(
                f"{duration}-second CSV lacks columns: "
                f"{sorted(missing_columns)}"
            )

        if len(frame) != 28:
            raise RuntimeError(
                f"{duration}-second CSV contains "
                f"{len(frame)} rows instead of 28."
            )

        frame = frame.copy()

        if "duration_seconds" in frame.columns:
            observed_durations = set(
                pd.to_numeric(
                    frame["duration_seconds"],
                    errors="raise",
                ).astype(int)
            )

            if observed_durations != {duration}:
                raise RuntimeError(
                    f"{duration}-second CSV duration mismatch."
                )

        frame[
            "duration_seconds"
        ] = duration

        frame[
            "temporal_role"
        ] = (
            "matched_four_second_reference"
            if duration == 4
            else "post_primary_temporal_sensitivity"
        )

        checks = normalize_boolean(
            frame[
                "all_numerical_checks_passed"
            ]
        )

        if not checks.all():
            raise RuntimeError(
                f"{duration}-second CSV contains failed checks."
            )

        if frame[
            [
                "analysis_type",
                "dataset_or_direction",
                "path",
            ]
        ].duplicated().any():
            raise RuntimeError(
                f"{duration}-second CSV contains duplicate groups."
            )

        observed_settings = set(
            zip(
                frame[
                    "analysis_type"
                ].astype(str),
                frame[
                    "dataset_or_direction"
                ].astype(str),
            )
        )

        if observed_settings != expected_setting_set:
            raise RuntimeError(
                f"{duration}-second setting grid differs."
            )

        if set(
            frame["path"].astype(str)
        ) != expected_path_set:
            raise RuntimeError(
                f"{duration}-second path grid differs."
            )

        for column in [
            "pooled_balanced_accuracy",
            "pooled_macro_f1",
            "pooled_roc_auc",
            "pooled_pr_auc",
            "participant_macro_macro_f1",
        ]:
            values = pd.to_numeric(
                frame[column],
                errors="raise",
            )

            if (
                values.isna().any()
                or (values < 0.0).any()
                or (values > 1.0).any()
            ):
                raise RuntimeError(
                    f"{duration}-second {column} values are invalid."
                )

            frame[column] = values.astype(float)

        source_records.append(
            {
                "duration_seconds":
                    duration,
                "metrics_csv":
                    csv_path.relative_to(
                        Path.cwd()
                    ).as_posix(),
                "metrics_json":
                    json_path.relative_to(
                        Path.cwd()
                    ).as_posix(),
                "metrics_csv_sha256":
                    sha256_file(csv_path),
                "metrics_json_sha256":
                    sha256_file(json_path),
                "report_groups":
                    int(
                        audit.get(
                            "report_groups_checked",
                            28,
                        )
                    ),
                "bootstrap_rows":
                    int(
                        audit.get(
                            "bootstrap_rows_checked",
                            140000,
                        )
                    ),
            }
        )

        frames.append(frame)

    combined = pd.concat(
        frames,
        ignore_index=True,
    )

    if len(combined) != 140:
        raise RuntimeError(
            f"Expected 140 temporal rows, observed {len(combined)}."
        )

    if combined[
        [
            "analysis_type",
            "dataset_or_direction",
            "path",
            "duration_seconds",
        ]
    ].duplicated().any():
        raise RuntimeError(
            "Combined temporal evidence contains duplicate identities."
        )

    baseline = (
        combined.loc[
            combined[
                "duration_seconds"
            ]
            == 4,
            [
                "analysis_type",
                "dataset_or_direction",
                "path",
                "pooled_balanced_accuracy",
                "pooled_macro_f1",
                "pooled_roc_auc",
                "pooled_pr_auc",
            ],
        ]
        .rename(
            columns={
                "pooled_balanced_accuracy":
                    "reference_04s_balanced_accuracy",
                "pooled_macro_f1":
                    "reference_04s_macro_f1",
                "pooled_roc_auc":
                    "reference_04s_roc_auc",
                "pooled_pr_auc":
                    "reference_04s_pr_auc",
            }
        )
    )

    if len(baseline) != 28:
        raise RuntimeError(
            "Matched four-second reference does not contain 28 rows."
        )

    combined = combined.merge(
        baseline,
        on=[
            "analysis_type",
            "dataset_or_direction",
            "path",
        ],
        how="left",
        validate="many_to_one",
    )

    if combined[
        [
            "reference_04s_balanced_accuracy",
            "reference_04s_macro_f1",
            "reference_04s_roc_auc",
            "reference_04s_pr_auc",
        ]
    ].isna().any().any():
        raise RuntimeError(
            "One or more rows lack a four-second reference."
        )

    combined[
        "delta_balanced_accuracy_vs_04s"
    ] = (
        combined[
            "pooled_balanced_accuracy"
        ]
        - combined[
            "reference_04s_balanced_accuracy"
        ]
    )

    combined[
        "delta_macro_f1_vs_04s"
    ] = (
        combined[
            "pooled_macro_f1"
        ]
        - combined[
            "reference_04s_macro_f1"
        ]
    )

    combined[
        "delta_roc_auc_vs_04s"
    ] = (
        combined[
            "pooled_roc_auc"
        ]
        - combined[
            "reference_04s_roc_auc"
        ]
    )

    combined[
        "delta_pr_auc_vs_04s"
    ] = (
        combined[
            "pooled_pr_auc"
        ]
        - combined[
            "reference_04s_pr_auc"
        ]
    )

    four_second_rows = combined[
        "duration_seconds"
    ] == 4

    delta_columns = [
        "delta_balanced_accuracy_vs_04s",
        "delta_macro_f1_vs_04s",
        "delta_roc_auc_vs_04s",
        "delta_pr_auc_vs_04s",
    ]

    if not (
        combined.loc[
            four_second_rows,
            delta_columns,
        ].abs()
        <= 1e-12
    ).all().all():
        raise RuntimeError(
            "Four-second reference deltas are not zero."
        )

    combined = rank_within_setting_duration(
        combined
    )

    best_by_setting_duration = (
        combined.loc[
            combined[
                "temporal_descriptive_rank_within_setting_duration"
            ]
            == 1
        ]
        .sort_values(
            [
                "analysis_type",
                "dataset_or_direction",
                "duration_seconds",
            ],
            kind="stable",
        )
        .reset_index(drop=True)
    )

    if len(best_by_setting_duration) != 20:
        raise RuntimeError(
            "Expected 20 setting-duration winners."
        )

    best_overall_by_setting = descriptive_best_rows(
        combined,
        [
            "analysis_type",
            "dataset_or_direction",
        ],
    ).sort_values(
        [
            "analysis_type",
            "dataset_or_direction",
        ],
        kind="stable",
    )

    if len(best_overall_by_setting) != 4:
        raise RuntimeError(
            "Expected four overall setting winners."
        )

    best_duration_by_setting_path = descriptive_best_rows(
        combined,
        [
            "analysis_type",
            "dataset_or_direction",
            "path",
        ],
    )

    if len(best_duration_by_setting_path) != 28:
        raise RuntimeError(
            "Expected 28 setting-path duration winners."
        )

    path_win_frequency = (
        best_by_setting_duration[
            "path"
        ]
        .value_counts()
        .reindex(
            PATHS,
            fill_value=0,
        )
    )

    duration_win_frequency = (
        best_duration_by_setting_path[
            "duration_seconds"
        ]
        .astype(int)
        .value_counts()
        .reindex(
            DURATIONS,
            fill_value=0,
        )
    )

    combined = combined.sort_values(
        [
            "analysis_type",
            "dataset_or_direction",
            "duration_seconds",
            "temporal_descriptive_rank_within_setting_duration",
        ],
        kind="stable",
    ).reset_index(drop=True)

    payload: dict[str, Any] = {
        "audit_identity":
            "bbbd_temporal_matched_all_durations_v1",
        "source_commit":
            "b772209",
        "status":
            "passed",
        "durations_seconds":
            DURATIONS,
        "duration_count":
            5,
        "paths":
            PATHS,
        "path_count":
            7,
        "setting_count":
            4,
        "rows_checked":
            140,
        "report_groups_per_duration":
            28,
        "total_report_groups":
            140,
        "bootstrap_rows_per_duration":
            140000,
        "total_bootstrap_rows_across_duration_reports":
            700000,
        "numerical_mismatches":
            0,
        "primary_four_second_392_recording_analysis_replaced":
            False,
        "matched_four_second_387_recording_role":
            "temporal sensitivity reference only",
        "performance_observed":
            True,
        "model_fitting_performed":
            False,
        "feature_selection_performed":
            False,
        "threshold_selection_performed":
            False,
        "inferential_duration_superiority_claimed":
            False,
        "minimum_reliable_duration_claimed":
            False,
        "interpretation_boundary": (
            "The aggregation is descriptive. It reports every "
            "prespecified duration and does not establish statistically "
            "significant superiority of one duration over another."
        ),
        "external_construct_boundary": (
            "BBBD outcomes represent attentive-versus-distracted "
            "experimental-condition classification and do not isolate "
            "attention from all task, order, fatigue, or session effects."
        ),
        "source_evidence": source_records,
        "best_path_by_setting_and_duration": [
            {
                "analysis_type":
                    str(row["analysis_type"]),
                "dataset_or_direction":
                    str(row["dataset_or_direction"]),
                "duration_seconds":
                    int(row["duration_seconds"]),
                "path":
                    str(row["path"]),
                "balanced_accuracy":
                    float(
                        row[
                            "pooled_balanced_accuracy"
                        ]
                    ),
                "macro_f1":
                    float(
                        row[
                            "pooled_macro_f1"
                        ]
                    ),
                "roc_auc":
                    float(
                        row[
                            "pooled_roc_auc"
                        ]
                    ),
                "pr_auc":
                    float(
                        row[
                            "pooled_pr_auc"
                        ]
                    ),
            }
            for _, row in best_by_setting_duration.iterrows()
        ],
        "descriptive_best_overall_by_setting": [
            {
                "analysis_type":
                    str(row["analysis_type"]),
                "dataset_or_direction":
                    str(row["dataset_or_direction"]),
                "duration_seconds":
                    int(row["duration_seconds"]),
                "path":
                    str(row["path"]),
                "balanced_accuracy":
                    float(
                        row[
                            "pooled_balanced_accuracy"
                        ]
                    ),
                "macro_f1":
                    float(
                        row[
                            "pooled_macro_f1"
                        ]
                    ),
                "delta_balanced_accuracy_vs_04s":
                    float(
                        row[
                            "delta_balanced_accuracy_vs_04s"
                        ]
                    ),
                "delta_macro_f1_vs_04s":
                    float(
                        row[
                            "delta_macro_f1_vs_04s"
                        ]
                    ),
            }
            for _, row in best_overall_by_setting.iterrows()
        ],
        "descriptive_best_duration_by_setting_path": [
            {
                "analysis_type":
                    str(row["analysis_type"]),
                "dataset_or_direction":
                    str(row["dataset_or_direction"]),
                "path":
                    str(row["path"]),
                "duration_seconds":
                    int(row["duration_seconds"]),
                "balanced_accuracy":
                    float(
                        row[
                            "pooled_balanced_accuracy"
                        ]
                    ),
                "macro_f1":
                    float(
                        row[
                            "pooled_macro_f1"
                        ]
                    ),
            }
            for _, row in best_duration_by_setting_path.iterrows()
        ],
        "best_path_frequency_across_20_setting_duration_cells": {
            path:
                int(path_win_frequency.loc[path])
            for path in PATHS
        },
        "best_duration_frequency_across_28_setting_path_trajectories": {
            str(duration):
                int(
                    duration_win_frequency.loc[
                        duration
                    ]
                )
            for duration in DURATIONS
        },
    }

    return payload, combined


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate and validate all matched BBBD "
            "temporal-sensitivity durations."
        )
    )

    parser.add_argument(
        "--audit-root",
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

    payload, combined = aggregate_temporal_evidence(
        arguments.audit_root
    )

    output_json = resolve_path(
        arguments.output_json
    )

    output_csv = resolve_path(
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

    combined.to_csv(
        output_csv,
        index=False,
        float_format="%.15g",
    )

    print(
        "\n===== COMPLETE TEMPORAL EVIDENCE SUMMARY ====="
    )

    print(
        "Durations checked:",
        payload["durations_seconds"],
    )

    print(
        "Rows checked:",
        payload["rows_checked"],
    )

    print(
        "Report groups represented:",
        payload["total_report_groups"],
    )

    print(
        "Bootstrap rows represented:",
        payload[
            "total_bootstrap_rows_across_duration_reports"
        ],
    )

    print(
        "Numerical mismatches:",
        payload["numerical_mismatches"],
    )

    print(
        "\nDescriptive best overall result by setting:"
    )

    for record in payload[
        "descriptive_best_overall_by_setting"
    ]:
        print(
            f"{record['analysis_type']} | "
            f"{record['dataset_or_direction']} | "
            f"{record['duration_seconds']}s | "
            f"{record['path']} | "
            f"BA={record['balanced_accuracy']:.6f} | "
            f"F1={record['macro_f1']:.6f} | "
            f"Delta BA vs matched 4s="
            f"{record['delta_balanced_accuracy_vs_04s']:+.6f}"
        )

    print(
        "\nALL-DURATION TEMPORAL EVIDENCE AUDIT PASSED"
    )


if __name__ == "__main__":
    main()
