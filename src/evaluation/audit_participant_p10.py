"""Objective participant-level quality and distribution audit.

This module investigates whether P10 has an objective technical defect that
could justify exclusion. It does not modify labels, features, predictions,
models, or reported results.

Hard exclusion-supporting failures are restricted to:

- missing expected recordings;
- incorrect EEG channel schema;
- unexpected recording-window counts;
- nonfinite ECG or EEG samples;
- excessive objectively flat ECG or EEG windows.

Feature-space distance and temporal drift are reported as distribution-shift
diagnostics only. They do not independently justify participant exclusion.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.core.config import load_revision_config
from src.features.internal_xr_long_window_features import (
    duration_directory_name,
    model_feature_columns,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_project_path(
    value: str,
) -> Path:
    """Resolve a project-relative or absolute path."""

    path = Path(value)

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path.resolve()


def robust_location_scale(
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return robust center and nonzero scale for feature diagnostics."""

    array = np.asarray(
        values,
        dtype=float,
    )

    center = np.median(
        array,
        axis=0,
    )

    first_quartile = np.percentile(
        array,
        25,
        axis=0,
    )

    third_quartile = np.percentile(
        array,
        75,
        axis=0,
    )

    interquartile_range = (
        third_quartile
        - first_quartile
    )

    standard_deviation = np.std(
        array,
        axis=0,
    )

    scale = np.where(
        interquartile_range > 1e-12,
        interquartile_range,
        np.where(
            standard_deviation > 1e-12,
            standard_deviation,
            1.0,
        ),
    )

    return center, scale


def robust_scalar_zscore(
    value: float,
    reference: np.ndarray,
) -> float:
    """Compute a robust scalar z-score using median absolute deviation."""

    reference = np.asarray(
        reference,
        dtype=float,
    )

    median = float(
        np.median(reference)
    )

    median_absolute_deviation = float(
        np.median(
            np.abs(
                reference - median
            )
        )
    )

    scale = (
        1.4826
        * median_absolute_deviation
    )

    if scale <= 1e-12:
        scale = float(
            np.std(
                reference,
                ddof=0,
            )
        )

    if scale <= 1e-12:
        return 0.0

    return float(
        (
            value - median
        )
        / scale
    )


def percentile_rank(
    value: float,
    reference: np.ndarray,
) -> float:
    """Return the empirical percentile rank in [0, 1]."""

    reference = np.asarray(
        reference,
        dtype=float,
    )

    return float(
        np.mean(
            reference <= value
        )
    )


def collect_raw_signal_quality(
    source_directory: Path,
    manifest: pd.DataFrame,
    expected_channels: list[str],
    flat_threshold: float,
) -> pd.DataFrame:
    """Audit corrected ECG and EEG archives for every participant and phase."""

    rows: list[dict[str, Any]] = []

    grouped = manifest.groupby(
        "recording_archive",
        sort=True,
    )

    for relative_archive, recording_manifest in grouped:
        participants = recording_manifest[
            "participant"
        ].astype(str).unique()

        phases = recording_manifest[
            "phase"
        ].astype(int).unique()

        labels = recording_manifest[
            "label"
        ].astype(int).unique()

        if (
            len(participants) != 1
            or len(phases) != 1
            or len(labels) != 1
        ):
            raise RuntimeError(
                f"Inconsistent recording metadata: {relative_archive}"
            )

        participant = str(
            participants[0]
        )

        phase = int(
            phases[0]
        )

        label = int(
            labels[0]
        )

        archive_path = (
            source_directory
            / str(relative_archive)
        )

        if not archive_path.is_file():
            rows.append(
                {
                    "participant": participant,
                    "phase": phase,
                    "label": label,
                    "archive": str(relative_archive),
                    "archive_exists": False,
                    "window_count": 0,
                    "manifest_window_count":
                        int(len(recording_manifest)),
                    "channel_schema_correct": False,
                    "nonfinite_eeg_samples": np.nan,
                    "nonfinite_ecg_samples": np.nan,
                    "maximum_eeg_flat_window_fraction": np.nan,
                    "ecg_flat_window_fraction": np.nan,
                    "median_eeg_absolute_amplitude": np.nan,
                    "median_ecg_absolute_amplitude": np.nan,
                    "median_eeg_window_peak_to_peak": np.nan,
                    "median_ecg_window_peak_to_peak": np.nan,
                }
            )

            continue

        with np.load(
            archive_path,
            allow_pickle=False,
        ) as archive:
            eeg = np.asarray(
                archive["eeg"],
                dtype=float,
            )

            ecg = np.asarray(
                archive["ecg"],
                dtype=float,
            )

            channels = [
                str(value)
                for value in archive[
                    "eeg_channels"
                ].tolist()
            ]

        if eeg.ndim != 3:
            raise RuntimeError(
                f"Unexpected EEG shape in {archive_path}: {eeg.shape}"
            )

        if ecg.ndim != 2:
            raise RuntimeError(
                f"Unexpected ECG shape in {archive_path}: {ecg.shape}"
            )

        eeg_window_standard_deviation = np.std(
            eeg,
            axis=1,
        )

        ecg_window_standard_deviation = np.std(
            ecg,
            axis=1,
        )

        maximum_eeg_flat_fraction = float(
            np.max(
                np.mean(
                    eeg_window_standard_deviation
                    <= flat_threshold,
                    axis=0,
                )
            )
        )

        ecg_flat_fraction = float(
            np.mean(
                ecg_window_standard_deviation
                <= flat_threshold
            )
        )

        rows.append(
            {
                "participant": participant,
                "phase": phase,
                "label": label,
                "archive": str(relative_archive),
                "archive_exists": True,
                "window_count": int(
                    eeg.shape[0]
                ),
                "manifest_window_count":
                    int(len(recording_manifest)),
                "channel_schema_correct":
                    channels == expected_channels,
                "nonfinite_eeg_samples":
                    int(
                        (
                            ~np.isfinite(eeg)
                        ).sum()
                    ),
                "nonfinite_ecg_samples":
                    int(
                        (
                            ~np.isfinite(ecg)
                        ).sum()
                    ),
                "maximum_eeg_flat_window_fraction":
                    maximum_eeg_flat_fraction,
                "ecg_flat_window_fraction":
                    ecg_flat_fraction,
                "median_eeg_absolute_amplitude":
                    float(
                        np.median(
                            np.abs(eeg)
                        )
                    ),
                "median_ecg_absolute_amplitude":
                    float(
                        np.median(
                            np.abs(ecg)
                        )
                    ),
                "median_eeg_window_peak_to_peak":
                    float(
                        np.median(
                            np.ptp(
                                eeg,
                                axis=1,
                            )
                        )
                    ),
                "median_ecg_window_peak_to_peak":
                    float(
                        np.median(
                            np.ptp(
                                ecg,
                                axis=1,
                            )
                        )
                    ),
            }
        )

    return pd.DataFrame(rows)


def add_signal_cohort_diagnostics(
    raw_quality: pd.DataFrame,
) -> pd.DataFrame:
    """Add robust cohort-relative amplitude diagnostics."""

    result = raw_quality.copy()

    metrics = [
        "median_eeg_absolute_amplitude",
        "median_ecg_absolute_amplitude",
        "median_eeg_window_peak_to_peak",
        "median_ecg_window_peak_to_peak",
    ]

    for metric in metrics:
        result[
            f"{metric}_robust_z"
        ] = np.nan

        for phase, indices in result.groupby(
            "phase"
        ).groups.items():
            phase_indices = list(indices)

            values = result.loc[
                phase_indices,
                metric,
            ].to_numpy(dtype=float)

            result.loc[
                phase_indices,
                f"{metric}_robust_z",
            ] = [
                robust_scalar_zscore(
                    value,
                    values,
                )
                for value in values
            ]

    return result


def feature_space_diagnostics(
    feature_root: Path,
    durations: list[int],
    target_participant: str,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Measure cohort-relative class geometry and temporal drift."""

    centroid_rows: list[
        dict[str, Any]
    ] = []

    drift_rows: list[
        dict[str, Any]
    ] = []

    nearest_rows: list[
        dict[str, Any]
    ] = []

    for duration in durations:
        path = (
            feature_root
            / duration_directory_name(
                duration,
                0.50,
            )
            / "ECG_EEG_features.csv"
        )

        if not path.is_file():
            raise FileNotFoundError(
                f"Long-window feature table not found: {path}"
            )

        frame = pd.read_csv(
            path,
            low_memory=False,
        )

        features = model_feature_columns(
            frame
        )

        values = frame[
            features
        ].to_numpy(dtype=float)

        center, scale = robust_location_scale(
            values
        )

        standardized = (
            values - center
        ) / scale

        working = frame[
            [
                "participant",
                "phase",
                "label",
                "start_seconds",
                "long_window_index",
            ]
        ].copy()

        feature_columns = [
            f"diagnostic_feature_{index:03d}"
            for index in range(
                standardized.shape[1]
            )
        ]

        standardized_frame = pd.DataFrame(
            standardized,
            columns=feature_columns,
            index=working.index,
        )

        working = pd.concat(
            [
                working,
                standardized_frame,
            ],
            axis=1,
        )

        participants = sorted(
            working[
                "participant"
            ].astype(str).unique()
        )

        for participant in participants:
            participant_rows = working.loc[
                working[
                    "participant"
                ].astype(str)
                == participant
            ]

            labels = participant_rows[
                "label"
            ].to_numpy(dtype=int)

            predictions = np.zeros(
                len(participant_rows),
                dtype=int,
            )

            reference_centroids: dict[
                int,
                np.ndarray,
            ] = {}

            for label in (
                0,
                1,
                2,
            ):
                reference = working.loc[
                    (
                        working[
                            "participant"
                        ].astype(str)
                        != participant
                    )
                    & (
                        working[
                            "label"
                        ].astype(int)
                        == label
                    ),
                    feature_columns,
                ].to_numpy(dtype=float)

                reference_centroids[
                    label
                ] = np.mean(
                    reference,
                    axis=0,
                )

            participant_values = participant_rows[
                feature_columns
            ].to_numpy(dtype=float)

            distance_matrix = np.column_stack(
                [
                    np.mean(
                        (
                            participant_values
                            - reference_centroids[label]
                        )
                        ** 2,
                        axis=1,
                    )
                    for label in (
                        0,
                        1,
                        2,
                    )
                ]
            )

            predictions[:] = np.argmin(
                distance_matrix,
                axis=1,
            )

            nearest_rows.append(
                {
                    "duration_seconds":
                        duration,
                    "participant":
                        participant,
                    "row_count":
                        int(
                            len(
                                participant_rows
                            )
                        ),
                    "nearest_reference_centroid_accuracy":
                        float(
                            np.mean(
                                predictions
                                == labels
                            )
                        ),
                    "is_target_participant":
                        participant
                        == target_participant,
                }
            )

            for phase in (
                1,
                2,
                3,
            ):
                phase_rows = participant_rows.loc[
                    participant_rows[
                        "phase"
                    ].astype(int)
                    == phase
                ].sort_values(
                    "start_seconds"
                )

                if phase_rows.empty:
                    raise RuntimeError(
                        f"{participant}, phase {phase}, "
                        f"duration {duration}: no feature rows."
                    )

                phase_label = int(
                    phase_rows[
                        "label"
                    ].iloc[0]
                )

                participant_centroid = np.mean(
                    phase_rows[
                        feature_columns
                    ].to_numpy(dtype=float),
                    axis=0,
                )

                reference_same_class = working.loc[
                    (
                        working[
                            "participant"
                        ].astype(str)
                        != participant
                    )
                    & (
                        working[
                            "label"
                        ].astype(int)
                        == phase_label
                    ),
                    feature_columns,
                ].to_numpy(dtype=float)

                reference_centroid = np.mean(
                    reference_same_class,
                    axis=0,
                )

                same_class_distance = float(
                    np.mean(
                        (
                            participant_centroid
                            - reference_centroid
                        )
                        ** 2
                    )
                )

                centroid_rows.append(
                    {
                        "duration_seconds":
                            duration,
                        "participant":
                            participant,
                        "phase":
                            phase,
                        "label":
                            phase_label,
                        "same_class_centroid_distance":
                            same_class_distance,
                        "is_target_participant":
                            participant
                            == target_participant,
                    }
                )

                phase_values = phase_rows[
                    feature_columns
                ].to_numpy(dtype=float)

                third = max(
                    1,
                    len(phase_values)
                    // 3,
                )

                early_centroid = np.mean(
                    phase_values[
                        :third
                    ],
                    axis=0,
                )

                late_centroid = np.mean(
                    phase_values[
                        -third:
                    ],
                    axis=0,
                )

                temporal_drift = float(
                    np.mean(
                        (
                            late_centroid
                            - early_centroid
                        )
                        ** 2
                    )
                )

                drift_rows.append(
                    {
                        "duration_seconds":
                            duration,
                        "participant":
                            participant,
                        "phase":
                            phase,
                        "label":
                            phase_label,
                        "early_windows":
                            third,
                        "late_windows":
                            third,
                        "early_to_late_centroid_distance":
                            temporal_drift,
                        "is_target_participant":
                            participant
                            == target_participant,
                    }
                )

    centroid_table = pd.DataFrame(
        centroid_rows
    )

    drift_table = pd.DataFrame(
        drift_rows
    )

    nearest_table = pd.DataFrame(
        nearest_rows
    )

    for table, metric in (
        (
            centroid_table,
            "same_class_centroid_distance",
        ),
        (
            drift_table,
            "early_to_late_centroid_distance",
        ),
    ):
        table[
            f"{metric}_percentile_within_duration_phase"
        ] = np.nan

        table[
            f"{metric}_robust_z_within_duration_phase"
        ] = np.nan

        grouped = table.groupby(
            [
                "duration_seconds",
                "phase",
            ]
        )

        for _, indices in grouped.groups.items():
            indices = list(indices)

            values = table.loc[
                indices,
                metric,
            ].to_numpy(dtype=float)

            table.loc[
                indices,
                f"{metric}_percentile_within_duration_phase",
            ] = [
                percentile_rank(
                    value,
                    values,
                )
                for value in values
            ]

            table.loc[
                indices,
                f"{metric}_robust_z_within_duration_phase",
            ] = [
                robust_scalar_zscore(
                    value,
                    values,
                )
                for value in values
            ]

    return (
        centroid_table,
        drift_table,
        nearest_table,
    )


def evaluate_hard_failures(
    target_raw_quality: pd.DataFrame,
    expected_recording_count: int,
    maximum_flat_fraction: float,
) -> list[str]:
    """Return only objective technical exclusion criteria that failed."""

    failures: list[str] = []

    if len(
        target_raw_quality
    ) != expected_recording_count:
        failures.append(
            "unexpected_recording_count"
        )

    if not target_raw_quality[
        "archive_exists"
    ].astype(bool).all():
        failures.append(
            "missing_recording_archive"
        )

    if not target_raw_quality[
        "channel_schema_correct"
    ].astype(bool).all():
        failures.append(
            "incorrect_eeg_channel_schema"
        )

    if not (
        target_raw_quality[
            "window_count"
        ].astype(int)
        == target_raw_quality[
            "manifest_window_count"
        ].astype(int)
    ).all():
        failures.append(
            "recording_window_count_mismatch"
        )

    if (
        target_raw_quality[
            "nonfinite_eeg_samples"
        ].fillna(1).astype(int)
        > 0
    ).any():
        failures.append(
            "nonfinite_eeg_samples"
        )

    if (
        target_raw_quality[
            "nonfinite_ecg_samples"
        ].fillna(1).astype(int)
        > 0
    ).any():
        failures.append(
            "nonfinite_ecg_samples"
        )

    if (
        target_raw_quality[
            "maximum_eeg_flat_window_fraction"
        ].fillna(1.0).astype(float)
        > maximum_flat_fraction
    ).any():
        failures.append(
            "excessive_flat_eeg_windows"
        )

    if (
        target_raw_quality[
            "ecg_flat_window_fraction"
        ].fillna(1.0).astype(float)
        > maximum_flat_fraction
    ).any():
        failures.append(
            "excessive_flat_ecg_windows"
        )

    return failures


def run_participant_audit(
    config: dict[str, Any],
    *,
    write_outputs: bool,
) -> dict[str, Any]:
    """Run the objective P10 audit."""

    protocol = config[
        "training"
    ][
        "baseline_calibrated_loso"
    ]

    audit_config = protocol[
        "participant_audit"
    ]

    target_participant = str(
        audit_config[
            "target_participant"
        ]
    )

    flat_threshold = float(
        audit_config[
            "flat_signal_standard_deviation_threshold"
        ]
    )

    maximum_flat_fraction = float(
        audit_config[
            "maximum_flat_window_fraction"
        ]
    )

    output_directory = resolve_project_path(
        audit_config[
            "output_dir"
        ]
    )

    source_directory = resolve_project_path(
        config[
            "datasets"
        ][
            "internal_xr"
        ][
            "output_dir"
        ]
    )

    manifest_path = (
        source_directory
        / "window_manifest.csv"
    )

    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Corrected manifest not found: {manifest_path}"
        )

    manifest = pd.read_csv(
        manifest_path,
        low_memory=False,
    )

    expected_channels = list(
        config[
            "eeg"
        ][
            "internal_channels"
        ]
    )

    raw_quality = collect_raw_signal_quality(
        source_directory,
        manifest,
        expected_channels,
        flat_threshold,
    )

    raw_quality = add_signal_cohort_diagnostics(
        raw_quality
    )

    target_raw_quality = raw_quality.loc[
        raw_quality[
            "participant"
        ].astype(str)
        == target_participant
    ].copy()

    hard_failures = evaluate_hard_failures(
        target_raw_quality,
        expected_recording_count=3,
        maximum_flat_fraction=(
            maximum_flat_fraction
        ),
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

    durations = [
        int(value)
        for value in protocol[
            "durations_seconds"
        ]
    ]

    (
        centroid_diagnostics,
        temporal_drift,
        nearest_centroid,
    ) = feature_space_diagnostics(
        feature_root,
        durations,
        target_participant,
    )

    target_centroid = centroid_diagnostics.loc[
        centroid_diagnostics[
            "participant"
        ].astype(str)
        == target_participant
    ].copy()

    target_drift = temporal_drift.loc[
        temporal_drift[
            "participant"
        ].astype(str)
        == target_participant
    ].copy()

    target_nearest = nearest_centroid.loc[
        nearest_centroid[
            "participant"
        ].astype(str)
        == target_participant
    ].copy()

    result_path = resolve_project_path(
        protocol[
            "output_dir"
        ]
    ) / "outer_fold_metrics.csv"

    recorded_performance: dict[
        str,
        Any,
    ] = {}

    if result_path.is_file():
        result_table = pd.read_csv(
            result_path,
            low_memory=False,
        )

        target_result = result_table.loc[
            result_table[
                "test_participant"
            ].astype(str)
            == target_participant
        ]

        if len(target_result) == 1:
            row = target_result.iloc[0]

            recorded_performance = {
                "accuracy":
                    float(
                        row[
                            "accuracy"
                        ]
                    ),
                "balanced_accuracy":
                    float(
                        row[
                            "balanced_accuracy"
                        ]
                    ),
                "macro_f1":
                    float(
                        row[
                            "macro_f1"
                        ]
                    ),
                "macro_roc_auc_ovr":
                    float(
                        row[
                            "macro_roc_auc_ovr"
                        ]
                    ),
                "macro_pr_auc":
                    float(
                        row[
                            "macro_pr_auc"
                        ]
                    ),
            }

    signal_metric_summary: dict[
        str,
        Any,
    ] = {}

    signal_metrics = [
        "median_eeg_absolute_amplitude",
        "median_ecg_absolute_amplitude",
        "median_eeg_window_peak_to_peak",
        "median_ecg_window_peak_to_peak",
    ]

    for metric in signal_metrics:
        signal_metric_summary[
            metric
        ] = {
            "target_phase_values": {
                str(int(row.phase)):
                    float(
                        getattr(
                            row,
                            metric,
                        )
                    )
                for row
                in target_raw_quality.itertuples(
                    index=False
                )
            },
            "target_robust_z_values": {
                str(int(row.phase)):
                    float(
                        getattr(
                            row,
                            f"{metric}_robust_z",
                        )
                    )
                for row
                in target_raw_quality.itertuples(
                    index=False
                )
            },
        }

    summary = {
        "target_participant":
            target_participant,
        "audit_status":
            "passed",
        "objective_hard_failures":
            hard_failures,
        "objective_exclusion_supported":
            bool(
                hard_failures
            ),
        "exclusion_decision": (
            "Objective exclusion may be considered only after "
            "manual verification of the listed hard failures."
            if hard_failures
            else (
                "No objective technical defect was detected. "
                "Low predictive performance alone does not justify "
                "participant exclusion."
            )
        ),
        "expected_recordings":
            3,
        "observed_recordings":
            int(
                len(
                    target_raw_quality
                )
            ),
        "flat_signal_standard_deviation_threshold":
            flat_threshold,
        "maximum_allowed_flat_window_fraction":
            maximum_flat_fraction,
        "recorded_baseline_calibrated_loso_performance":
            recorded_performance,
        "signal_metric_summary":
            signal_metric_summary,
        "target_feature_distance_diagnostics":
            target_centroid.to_dict(
                orient="records"
            ),
        "target_temporal_drift_diagnostics":
            target_drift.to_dict(
                orient="records"
            ),
        "target_nearest_reference_centroid_accuracy":
            target_nearest.to_dict(
                orient="records"
            ),
        "interpretation": (
            "Large cohort-relative feature distance or temporal drift "
            "indicates distribution shift but is not evidence of data "
            "corruption and does not independently justify exclusion."
        ),
    }

    report_lines = [
        "# P10 Objective Data-Quality Audit",
        "",
        "## Exclusion verdict",
        "",
        (
            f"- Objective technical failures: "
            f"{', '.join(hard_failures) if hard_failures else 'none'}"
        ),
        (
            f"- Objective exclusion supported: "
            f"{bool(hard_failures)}"
        ),
        "",
        summary[
            "exclusion_decision"
        ],
        "",
        "## Recording integrity",
        "",
        (
            f"- Expected recordings: 3"
        ),
        (
            f"- Observed recordings: "
            f"{len(target_raw_quality)}"
        ),
        (
            f"- Nonfinite EEG samples: "
            f"{int(target_raw_quality['nonfinite_eeg_samples'].sum())}"
        ),
        (
            f"- Nonfinite ECG samples: "
            f"{int(target_raw_quality['nonfinite_ecg_samples'].sum())}"
        ),
        (
            f"- Maximum EEG flat-window fraction: "
            f"{target_raw_quality['maximum_eeg_flat_window_fraction'].max():.4f}"
        ),
        (
            f"- Maximum ECG flat-window fraction: "
            f"{target_raw_quality['ecg_flat_window_fraction'].max():.4f}"
        ),
        "",
        "## Distribution-shift interpretation",
        "",
        (
            "Feature-space distance and early-to-late temporal drift "
            "are diagnostic indicators only. They must not be used as "
            "post hoc participant-exclusion criteria."
        ),
        "",
    ]

    if write_outputs:
        temporary_directory = (
            output_directory.with_name(
                output_directory.name
                + ".building"
            )
        )

        if output_directory.exists():
            raise FileExistsError(
                "Refusing to overwrite existing participant audit: "
                f"{output_directory}"
            )

        if temporary_directory.exists():
            raise FileExistsError(
                "Temporary participant audit directory exists: "
                f"{temporary_directory}"
            )

        temporary_directory.mkdir(
            parents=True,
            exist_ok=False,
        )

        raw_quality.to_csv(
            temporary_directory
            / "all_participants_raw_signal_quality.csv",
            index=False,
        )

        target_raw_quality.to_csv(
            temporary_directory
            / "p10_raw_signal_quality.csv",
            index=False,
        )

        centroid_diagnostics.to_csv(
            temporary_directory
            / "all_participants_feature_distance.csv",
            index=False,
        )

        target_centroid.to_csv(
            temporary_directory
            / "p10_feature_distance.csv",
            index=False,
        )

        temporal_drift.to_csv(
            temporary_directory
            / "all_participants_temporal_drift.csv",
            index=False,
        )

        target_drift.to_csv(
            temporary_directory
            / "p10_temporal_drift.csv",
            index=False,
        )

        nearest_centroid.to_csv(
            temporary_directory
            / "nearest_reference_centroid_accuracy.csv",
            index=False,
        )

        (
            temporary_directory
            / "p10_audit_summary.json"
        ).write_text(
            json.dumps(
                summary,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        (
            temporary_directory
            / "p10_audit_report.md"
        ).write_text(
            "\n".join(
                report_lines
            ),
            encoding="utf-8",
        )

        temporary_directory.replace(
            output_directory
        )

    print(
        "\n===== P10 OBJECTIVE AUDIT ====="
    )

    print(
        "Objective hard failures:",
        hard_failures
        if hard_failures
        else "none",
    )

    print(
        "Objective exclusion supported:",
        bool(
            hard_failures
        ),
    )

    print(
        "Observed recordings:",
        len(
            target_raw_quality
        ),
    )

    print(
        "Nonfinite EEG samples:",
        int(
            target_raw_quality[
                "nonfinite_eeg_samples"
            ].sum()
        ),
    )

    print(
        "Nonfinite ECG samples:",
        int(
            target_raw_quality[
                "nonfinite_ecg_samples"
            ].sum()
        ),
    )

    print(
        "Maximum EEG flat-window fraction:",
        f"{target_raw_quality['maximum_eeg_flat_window_fraction'].max():.4f}",
    )

    print(
        "Maximum ECG flat-window fraction:",
        f"{target_raw_quality['ecg_flat_window_fraction'].max():.4f}",
    )

    print(
        "\nP10 nearest-reference-centroid diagnostic:"
    )

    print(
        target_nearest[
            [
                "duration_seconds",
                "row_count",
                "nearest_reference_centroid_accuracy",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\nP10 feature-distance percentiles:"
    )

    print(
        target_centroid[
            [
                "duration_seconds",
                "phase",
                "same_class_centroid_distance",
                (
                    "same_class_centroid_distance_"
                    "percentile_within_duration_phase"
                ),
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\nP10 temporal-drift percentiles:"
    )

    print(
        target_drift[
            [
                "duration_seconds",
                "phase",
                "early_to_late_centroid_distance",
                (
                    "early_to_late_centroid_distance_"
                    "percentile_within_duration_phase"
                ),
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\nFinal audit decision:"
    )

    print(
        summary[
            "exclusion_decision"
        ]
    )

    if write_outputs:
        print(
            "Audit output:",
            output_directory,
        )

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit P10 for objective technical exclusion criteria."
        )
    )

    parser.add_argument(
        "--write",
        action="store_true",
        help="Write complete participant-audit artifacts.",
    )

    arguments = parser.parse_args()

    config = load_revision_config(
        check_input_paths=True
    )

    run_participant_audit(
        config,
        write_outputs=arguments.write,
    )


if __name__ == "__main__":
    main()