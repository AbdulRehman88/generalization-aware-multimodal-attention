"""Independent numerical audit of matched BBBD four-second reports."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)


PATHS = (
    "EEG",
    "ECG",
    "Pupil",
    "ECG_EEG",
    "ECG_Pupil",
    "EEG_Pupil",
    "ECG_EEG_Pupil",
)

SETTINGS = (
    (
        "within_experiment",
        "experiment2",
        20,
        200,
    ),
    (
        "within_experiment",
        "experiment3",
        16,
        187,
    ),
    (
        "cross_experiment",
        "experiment2_to_experiment3",
        16,
        187,
    ),
    (
        "cross_experiment",
        "experiment3_to_experiment2",
        20,
        200,
    ),
)

METRICS = (
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "roc_auc",
    "pr_auc",
)

TOLERANCE = 1e-10


def resolve_project_path(value: str | Path) -> Path:
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


def assert_close(
    observed: float,
    expected: float,
    identity: str,
    tolerance: float = TOLERANCE,
) -> None:
    observed_value = float(observed)
    expected_value = float(expected)

    if not np.isfinite(observed_value):
        raise RuntimeError(
            f"{identity} observed value is nonfinite: {observed_value}"
        )

    if not np.isfinite(expected_value):
        raise RuntimeError(
            f"{identity} expected value is nonfinite: {expected_value}"
        )

    if not np.isclose(
        observed_value,
        expected_value,
        rtol=0.0,
        atol=tolerance,
    ):
        raise RuntimeError(
            f"{identity} mismatch: "
            f"{observed_value:.17g} versus {expected_value:.17g}"
        )


def compute_binary_metrics(
    labels: pd.Series,
    probabilities: pd.Series,
    predictions: pd.Series,
) -> dict[str, float]:
    y_true = labels.astype(int).to_numpy()
    y_probability = pd.to_numeric(
        probabilities,
        errors="raise",
    ).to_numpy(
        dtype=float
    )
    y_prediction = predictions.astype(int).to_numpy()

    if set(np.unique(y_true)) != {
        0,
        1,
    }:
        raise RuntimeError(
            "Metric group does not contain both binary labels."
        )

    if (
        not np.isfinite(y_probability).all()
        or (y_probability < 0.0).any()
        or (y_probability > 1.0).any()
    ):
        raise RuntimeError(
            "Metric group contains invalid probabilities."
        )

    return {
        "accuracy":
            float(
                accuracy_score(
                    y_true,
                    y_prediction,
                )
            ),
        "balanced_accuracy":
            float(
                balanced_accuracy_score(
                    y_true,
                    y_prediction,
                )
            ),
        "macro_f1":
            float(
                f1_score(
                    y_true,
                    y_prediction,
                    average="macro",
                    zero_division=0,
                )
            ),
        "roc_auc":
            float(
                roc_auc_score(
                    y_true,
                    y_probability,
                )
            ),
        "pr_auc":
            float(
                average_precision_score(
                    y_true,
                    y_probability,
                )
            ),
    }


def recompute_participant_metrics(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for participant, frame in predictions.groupby(
        "participant",
        sort=True,
    ):
        metrics = compute_binary_metrics(
            frame["label"],
            frame["probability"],
            frame["predicted_label"],
        )

        rows.append(
            {
                "participant":
                    str(participant),
                "recordings":
                    int(len(frame)),
                **metrics,
            }
        )

    return pd.DataFrame(
        rows
    ).sort_values(
        "participant",
        kind="stable",
    ).reset_index(
        drop=True
    )


def validate_confidence_intervals(
    bootstrap: pd.DataFrame,
    confidence_intervals: pd.DataFrame,
    identity: str,
) -> dict[str, dict[str, float]]:
    if len(bootstrap) != 5000:
        raise RuntimeError(
            f"{identity} contains {len(bootstrap)} bootstrap rows."
        )

    expected_repetitions = np.arange(
        5000,
        dtype=int,
    )

    observed_repetitions = bootstrap[
        "repetition"
    ].astype(int).to_numpy()

    if not np.array_equal(
        observed_repetitions,
        expected_repetitions,
    ):
        raise RuntimeError(
            f"{identity} bootstrap repetition sequence changed."
        )

    expected_metric_columns = [
        column
        for column in bootstrap.columns
        if column != "repetition"
    ]

    if set(
        confidence_intervals[
            "metric"
        ].astype(str)
    ) != set(expected_metric_columns):
        raise RuntimeError(
            f"{identity} confidence-interval metric set changed."
        )

    interval_lookup = (
        confidence_intervals
        .set_index(
            "metric"
        )
    )

    validated: dict[str, dict[str, float]] = {}

    alpha = 0.025

    for metric in expected_metric_columns:
        values = pd.to_numeric(
            bootstrap[metric],
            errors="coerce",
        )

        finite = values[
            np.isfinite(values)
        ].to_numpy(
            dtype=float
        )

        if len(finite) == 0:
            raise RuntimeError(
                f"{identity}/{metric} has no finite bootstrap values."
            )

        expected_mean = float(
            np.mean(finite)
        )

        expected_lower = float(
            np.quantile(
                finite,
                alpha,
                method="linear",
            )
        )

        expected_upper = float(
            np.quantile(
                finite,
                1.0 - alpha,
                method="linear",
            )
        )

        row = interval_lookup.loc[
            metric
        ]

        assert_close(
            row["bootstrap_mean"],
            expected_mean,
            f"{identity}/{metric}/bootstrap_mean",
        )

        assert_close(
            row["ci_lower"],
            expected_lower,
            f"{identity}/{metric}/ci_lower",
        )

        assert_close(
            row["ci_upper"],
            expected_upper,
            f"{identity}/{metric}/ci_upper",
        )

        if int(
            row[
                "finite_repetitions"
            ]
        ) != len(finite):
            raise RuntimeError(
                f"{identity}/{metric} finite repetition count changed."
            )

        validated[metric] = {
            "bootstrap_mean":
                expected_mean,
            "ci_lower":
                expected_lower,
            "ci_upper":
                expected_upper,
            "finite_repetitions":
                int(len(finite)),
        }

    return validated


def audit_reports(
    report_root: str | Path,
) -> tuple[
    dict[str, Any],
    pd.DataFrame,
]:
    resolved_root = resolve_project_path(
        report_root
    )

    if not resolved_root.is_dir():
        raise FileNotFoundError(
            resolved_root
        )

    manifest_path = (
        resolved_root
        / "reports_manifest.json"
    )

    registry_path = (
        resolved_root
        / "reports_registry.csv"
    )

    if not manifest_path.is_file():
        raise FileNotFoundError(
            manifest_path
        )

    if not registry_path.is_file():
        raise FileNotFoundError(
            registry_path
        )

    manifest = json.loads(
        manifest_path.read_text(
            encoding="utf-8"
        )
    )

    registry = pd.read_csv(
        registry_path,
        low_memory=False,
    )

    if manifest.get(
        "report_count"
    ) != 28:
        raise RuntimeError(
            "Report manifest does not contain 28 reports."
        )

    if len(registry) != 28:
        raise RuntimeError(
            "Report registry does not contain 28 rows."
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

    manifest_records = {
        (
            str(record["analysis_type"]),
            str(record["dataset_or_direction"]),
            str(record["path"]),
        ):
            record
        for record in manifest[
            "reports"
        ]
    }

    if len(manifest_records) != 28:
        raise RuntimeError(
            "Manifest report identities are not unique."
        )

    result_rows: list[dict[str, Any]] = []
    evidence_hashes: dict[str, str] = {}

    print(
        "\n===== RECOMPUTE ALL 28 REPORT GROUPS ====="
    )

    for (
        analysis_type,
        setting,
        expected_participants,
        expected_recordings,
    ) in SETTINGS:
        for path_name in PATHS:
            identity = (
                analysis_type,
                setting,
                path_name,
            )

            group_root = (
                resolved_root
                / analysis_type
                / setting
                / path_name
            )

            required_files = {
                "summary":
                    group_root
                    / "summary.json",
                "predictions":
                    group_root
                    / "recording_predictions.csv",
                "participant_metrics":
                    group_root
                    / "participant_metrics.csv",
                "bootstrap":
                    group_root
                    / "participant_bootstrap.csv",
                "confidence_intervals":
                    group_root
                    / "confidence_intervals.csv",
            }

            for file_path in required_files.values():
                if not file_path.is_file():
                    raise FileNotFoundError(
                        file_path
                    )

            summary = json.loads(
                required_files[
                    "summary"
                ].read_text(
                    encoding="utf-8"
                )
            )

            predictions = pd.read_csv(
                required_files[
                    "predictions"
                ],
                low_memory=False,
            )

            stored_participant_metrics = pd.read_csv(
                required_files[
                    "participant_metrics"
                ],
                low_memory=False,
            ).sort_values(
                "participant",
                kind="stable",
            ).reset_index(
                drop=True
            )

            bootstrap = pd.read_csv(
                required_files[
                    "bootstrap"
                ],
                low_memory=False,
            )

            confidence_intervals = pd.read_csv(
                required_files[
                    "confidence_intervals"
                ],
                low_memory=False,
            )

            if len(predictions) != expected_recordings:
                raise RuntimeError(
                    f"{identity} recording count differs: "
                    f"{len(predictions)} versus {expected_recordings}."
                )

            if predictions[
                "recording_id"
            ].duplicated().any():
                raise RuntimeError(
                    f"{identity} contains duplicate recording IDs."
                )

            if predictions[
                "participant"
            ].nunique() != expected_participants:
                raise RuntimeError(
                    f"{identity} participant count differs."
                )

            expected_prediction = (
                pd.to_numeric(
                    predictions[
                        "probability"
                    ],
                    errors="raise",
                )
                >= pd.to_numeric(
                    predictions[
                        "threshold"
                    ],
                    errors="raise",
                )
            ).astype(int)

            if not np.array_equal(
                expected_prediction.to_numpy(),
                predictions[
                    "predicted_label"
                ].astype(int).to_numpy(),
            ):
                raise RuntimeError(
                    f"{identity} contains a threshold-decision mismatch."
                )

            pooled = compute_binary_metrics(
                predictions[
                    "label"
                ],
                predictions[
                    "probability"
                ],
                predictions[
                    "predicted_label"
                ],
            )

            for metric in METRICS:
                assert_close(
                    pooled[metric],
                    summary[
                        "pooled_recording_metrics"
                    ][
                        metric
                    ],
                    (
                        f"{identity}/"
                        f"pooled_{metric}"
                    ),
                )

            recomputed_participant_metrics = (
                recompute_participant_metrics(
                    predictions
                )
            )

            if list(
                stored_participant_metrics[
                    "participant"
                ].astype(str)
            ) != list(
                recomputed_participant_metrics[
                    "participant"
                ].astype(str)
            ):
                raise RuntimeError(
                    f"{identity} participant ordering or membership changed."
                )

            if not np.array_equal(
                stored_participant_metrics[
                    "recordings"
                ].astype(int).to_numpy(),
                recomputed_participant_metrics[
                    "recordings"
                ].astype(int).to_numpy(),
            ):
                raise RuntimeError(
                    f"{identity} participant recording counts changed."
                )

            for metric in METRICS:
                stored_values = pd.to_numeric(
                    stored_participant_metrics[
                        metric
                    ],
                    errors="raise",
                ).to_numpy(
                    dtype=float
                )

                recomputed_values = pd.to_numeric(
                    recomputed_participant_metrics[
                        metric
                    ],
                    errors="raise",
                ).to_numpy(
                    dtype=float
                )

                if not np.allclose(
                    stored_values,
                    recomputed_values,
                    rtol=0.0,
                    atol=TOLERANCE,
                ):
                    raise RuntimeError(
                        f"{identity} participant {metric} values changed."
                    )

            participant_macro = {
                metric:
                    float(
                        recomputed_participant_metrics[
                            metric
                        ].mean()
                    )
                for metric in METRICS
            }

            for metric in METRICS:
                assert_close(
                    participant_macro[
                        metric
                    ],
                    summary[
                        "participant_macro_metrics"
                    ][
                        metric
                    ],
                    (
                        f"{identity}/"
                        f"participant_macro_{metric}"
                    ),
                )

            assert_close(
                expected_participants,
                summary[
                    "participant_macro_metrics"
                ][
                    "participant_count"
                ],
                (
                    f"{identity}/"
                    "participant_macro_count"
                ),
            )

            if int(
                summary[
                    "participants"
                ]
            ) != expected_participants:
                raise RuntimeError(
                    f"{identity} summary participant count changed."
                )

            if int(
                summary[
                    "recordings"
                ]
            ) != expected_recordings:
                raise RuntimeError(
                    f"{identity} summary recording count changed."
                )

            observed_threshold_counts = {
                str(float(key)):
                    int(value)
                for key, value in (
                    predictions[
                        "threshold"
                    ]
                    .value_counts(
                        sort=False
                    )
                    .sort_index()
                    .items()
                )
            }

            stored_threshold_counts = {
                str(float(key)):
                    int(value)
                for key, value in summary[
                    "threshold_counts"
                ].items()
            }

            if (
                observed_threshold_counts
                != stored_threshold_counts
            ):
                raise RuntimeError(
                    f"{identity} threshold counts changed."
                )

            intervals = validate_confidence_intervals(
                bootstrap,
                confidence_intervals,
                "::".join(
                    identity
                ),
            )

            registry_row = registry.loc[
                (
                    registry[
                        "analysis_type"
                    ].astype(str)
                    == analysis_type
                )
                & (
                    registry[
                        "dataset_or_direction"
                    ].astype(str)
                    == setting
                )
                & (
                    registry[
                        "path"
                    ].astype(str)
                    == path_name
                )
            ]

            if len(registry_row) != 1:
                raise RuntimeError(
                    f"{identity} registry row is missing or duplicated."
                )

            registry_record = registry_row.iloc[
                0
            ]

            manifest_record = manifest_records.get(
                identity
            )

            if manifest_record is None:
                raise RuntimeError(
                    f"{identity} manifest entry is missing."
                )

            summary_hash = sha256_file(
                required_files[
                    "summary"
                ]
            )

            bootstrap_hash = sha256_file(
                required_files[
                    "bootstrap"
                ]
            )

            if str(
                registry_record[
                    "summary_sha256"
                ]
            ) != summary_hash:
                raise RuntimeError(
                    f"{identity} registry summary hash mismatch."
                )

            if str(
                registry_record[
                    "bootstrap_sha256"
                ]
            ) != bootstrap_hash:
                raise RuntimeError(
                    f"{identity} registry bootstrap hash mismatch."
                )

            if str(
                manifest_record[
                    "summary_sha256"
                ]
            ) != summary_hash:
                raise RuntimeError(
                    f"{identity} manifest summary hash mismatch."
                )

            if str(
                manifest_record[
                    "bootstrap_sha256"
                ]
            ) != bootstrap_hash:
                raise RuntimeError(
                    f"{identity} manifest bootstrap hash mismatch."
                )

            evidence_hashes[
                required_files[
                    "summary"
                ].relative_to(
                    resolved_root
                ).as_posix()
            ] = summary_hash

            evidence_hashes[
                required_files[
                    "bootstrap"
                ].relative_to(
                    resolved_root
                ).as_posix()
            ] = bootstrap_hash

            result_rows.append(
                {
                    "duration_seconds":
                        4,
                    "scientific_role":
                        "matched_sensitivity_reference",
                    "analysis_type":
                        analysis_type,
                    "dataset_or_direction":
                        setting,
                    "path":
                        path_name,
                    "participants":
                        expected_participants,
                    "recordings":
                        expected_recordings,
                    "pooled_accuracy":
                        pooled[
                            "accuracy"
                        ],
                    "pooled_balanced_accuracy":
                        pooled[
                            "balanced_accuracy"
                        ],
                    "pooled_macro_f1":
                        pooled[
                            "macro_f1"
                        ],
                    "pooled_roc_auc":
                        pooled[
                            "roc_auc"
                        ],
                    "pooled_pr_auc":
                        pooled[
                            "pr_auc"
                        ],
                    "participant_macro_accuracy":
                        participant_macro[
                            "accuracy"
                        ],
                    "participant_macro_balanced_accuracy":
                        participant_macro[
                            "balanced_accuracy"
                        ],
                    "participant_macro_macro_f1":
                        participant_macro[
                            "macro_f1"
                        ],
                    "participant_macro_roc_auc":
                        participant_macro[
                            "roc_auc"
                        ],
                    "participant_macro_pr_auc":
                        participant_macro[
                            "pr_auc"
                        ],
                    "pooled_balanced_accuracy_ci_lower":
                        intervals[
                            "pooled_balanced_accuracy"
                        ][
                            "ci_lower"
                        ],
                    "pooled_balanced_accuracy_ci_upper":
                        intervals[
                            "pooled_balanced_accuracy"
                        ][
                            "ci_upper"
                        ],
                    "pooled_macro_f1_ci_lower":
                        intervals[
                            "pooled_macro_f1"
                        ][
                            "ci_lower"
                        ],
                    "pooled_macro_f1_ci_upper":
                        intervals[
                            "pooled_macro_f1"
                        ][
                            "ci_upper"
                        ],
                    "participant_macro_balanced_accuracy_ci_lower":
                        intervals[
                            "participant_macro_balanced_accuracy"
                        ][
                            "ci_lower"
                        ],
                    "participant_macro_balanced_accuracy_ci_upper":
                        intervals[
                            "participant_macro_balanced_accuracy"
                        ][
                            "ci_upper"
                        ],
                    "participant_macro_macro_f1_ci_lower":
                        intervals[
                            "participant_macro_macro_f1"
                        ][
                            "ci_lower"
                        ],
                    "participant_macro_macro_f1_ci_upper":
                        intervals[
                            "participant_macro_macro_f1"
                        ][
                            "ci_upper"
                        ],
                    "threshold_values":
                        ",".join(
                            sorted(
                                stored_threshold_counts,
                                key=float,
                            )
                        ),
                    "summary_sha256":
                        summary_hash,
                    "bootstrap_sha256":
                        bootstrap_hash,
                    "all_numerical_checks_passed":
                        True,
                }
            )

            print(
                f"{analysis_type:<18} | "
                f"{setting:<27} | "
                f"{path_name:<14} | "
                f"BA={pooled['balanced_accuracy']:.6f} | "
                f"F1={pooled['macro_f1']:.6f}"
            )

    metrics = pd.DataFrame(
        result_rows
    )

    if len(metrics) != 28:
        raise RuntimeError(
            f"Expected 28 metric rows, observed {len(metrics)}."
        )

    metrics[
        "descriptive_rank_within_setting"
    ] = (
        metrics
        .groupby(
            [
                "analysis_type",
                "dataset_or_direction",
            ],
            sort=False,
            group_keys=False,
        )
        .apply(
            lambda frame: (
                frame.sort_values(
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
                .assign(
                    descriptive_rank_within_setting=
                        range(
                            1,
                            len(frame) + 1,
                        )
                )[
                    "descriptive_rank_within_setting"
                ]
            ),
            include_groups=False,
        )
        .sort_index()
        .astype(int)
    )

    metrics = metrics.sort_values(
        [
            "analysis_type",
            "dataset_or_direction",
            "descriptive_rank_within_setting",
        ],
        kind="stable",
    ).reset_index(
        drop=True
    )

    best_rows = (
        metrics.loc[
            metrics[
                "descriptive_rank_within_setting"
            ] == 1
        ]
        .copy()
        .sort_values(
            [
                "analysis_type",
                "dataset_or_direction",
            ],
            kind="stable",
        )
    )

    payload = {
        "audit_identity":
            "bbbd_temporal_matched_04s_metrics_v1",
        "source_commit":
            "0695291",
        "status":
            "passed",
        "duration_seconds":
            4,
        "scientific_role":
            "matched sensitivity reference only",
        "primary_four_second_392_recording_analysis_replaced":
            False,
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
        "pooled_metrics_recomputed":
            True,
        "participant_metrics_recomputed":
            True,
        "participant_macro_metrics_recomputed":
            True,
        "threshold_decisions_recomputed":
            True,
        "confidence_intervals_recomputed":
            True,
        "manifest_hashes_verified":
            True,
        "registry_hashes_verified":
            True,
        "numerical_mismatches":
            0,
        "performance_observed":
            True,
        "interpretation_boundary":
            (
                "BBBD attentive-versus-distracted experimental-condition "
                "classification; not isolated attention-state recognition."
            ),
        "descriptive_ranking_policy":
            (
                "Balanced accuracy, then macro-F1, then ROC-AUC, then "
                "alphabetical path name. Ranking is descriptive only and "
                "does not alter model selection or the prespecified protocol."
            ),
        "best_path_by_setting": [
            {
                "analysis_type":
                    str(row[
                        "analysis_type"
                    ]),
                "dataset_or_direction":
                    str(row[
                        "dataset_or_direction"
                    ]),
                "path":
                    str(row[
                        "path"
                    ]),
                "pooled_balanced_accuracy":
                    float(
                        row[
                            "pooled_balanced_accuracy"
                        ]
                    ),
                "pooled_macro_f1":
                    float(
                        row[
                            "pooled_macro_f1"
                        ]
                    ),
                "pooled_roc_auc":
                    float(
                        row[
                            "pooled_roc_auc"
                        ]
                    ),
                "pooled_pr_auc":
                    float(
                        row[
                            "pooled_pr_auc"
                        ]
                    ),
            }
            for _, row in best_rows.iterrows()
        ],
        "reports_manifest_sha256":
            sha256_file(
                manifest_path
            ),
        "reports_registry_sha256":
            sha256_file(
                registry_path
            ),
        "evidence_hashes":
            evidence_hashes,
    }

    return payload, metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Independently audit the matched BBBD four-second reports."
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

    payload, metrics = audit_reports(
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
        "\n===== NUMERICAL AUDIT SUMMARY ====="
    )

    print(
        "Report groups checked:",
        payload[
            "report_groups_checked"
        ],
    )

    print(
        "Bootstrap rows checked:",
        payload[
            "bootstrap_rows_checked"
        ],
    )

    print(
        "Numerical mismatches:",
        payload[
            "numerical_mismatches"
        ],
    )

    print(
        "\nDescriptive best path by setting:"
    )

    for record in payload[
        "best_path_by_setting"
    ]:
        print(
            f"{record['analysis_type']} | "
            f"{record['dataset_or_direction']} | "
            f"{record['path']} | "
            f"BA={record['pooled_balanced_accuracy']:.6f} | "
            f"F1={record['pooled_macro_f1']:.6f}"
        )

    print(
        "\nINDEPENDENT 4-SECOND NUMERICAL AUDIT PASSED"
    )


if __name__ == "__main__":
    main()
