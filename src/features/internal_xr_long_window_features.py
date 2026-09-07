"""Longer, exactly 50%-overlapping internal XR feature generation.

The source consists of the corrected four-second archives produced after
full-recording filtering, resampling, channel restriction, and pupil-quality
processing. Consecutive corrected blocks are concatenated to construct longer
windows without repeating or altering signal preprocessing.

Supported exact designs on the four-second source grid:

- 8-second window, 4-second stride
- 16-second window, 8-second stride
- 24-second window, 12-second stride
- 32-second window, 16-second stride

No participant splitting, normalization, feature cleanup, feature selection,
model fitting, or evaluation occurs in this module.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.core.config import load_revision_config
from src.features.internal_xr_revision_features import (
    extract_ecg_features,
    extract_eeg_features,
    extract_pupil_features,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

METADATA_COLUMNS = [
    "segment_id",
    "participant",
    "phase",
    "label",
    "long_window_index",
    "start_base_window",
    "end_base_window_exclusive",
    "start_seconds",
    "end_seconds",
    "window_seconds",
    "stride_seconds",
    "overlap_fraction",
    "pupil_available",
]


@dataclass(frozen=True)
class LongWindowPlan:
    """Exact longer-window construction plan."""

    duration_seconds: int
    stride_seconds: int
    source_blocks_per_window: int
    source_blocks_per_stride: int
    start_indices: tuple[int, ...]


def resolve_project_path(
    value: str,
) -> Path:
    """Resolve an absolute or project-relative path."""

    path = Path(value)

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path.resolve()


def build_long_window_plan(
    recording_base_windows: int,
    *,
    source_window_seconds: int,
    duration_seconds: int,
    overlap_fraction: float,
) -> LongWindowPlan:
    """Create an exact source-block window plan."""

    if recording_base_windows <= 0:
        raise ValueError(
            "recording_base_windows must be positive."
        )

    if source_window_seconds <= 0:
        raise ValueError(
            "source_window_seconds must be positive."
        )

    if duration_seconds <= source_window_seconds:
        raise ValueError(
            "Long-window duration must exceed the source duration."
        )

    if not 0.0 <= overlap_fraction < 1.0:
        raise ValueError(
            "overlap_fraction must be in [0, 1)."
        )

    block_ratio = (
        duration_seconds
        / source_window_seconds
    )

    if not float(block_ratio).is_integer():
        raise ValueError(
            f"{duration_seconds}-second windows cannot be represented "
            f"exactly using {source_window_seconds}-second blocks."
        )

    source_blocks_per_window = int(
        block_ratio
    )

    stride_seconds_float = (
        duration_seconds
        * (
            1.0
            - overlap_fraction
        )
    )

    stride_block_ratio = (
        stride_seconds_float
        / source_window_seconds
    )

    if not float(
        stride_block_ratio
    ).is_integer():
        raise ValueError(
            f"{duration_seconds}-second windows with "
            f"{overlap_fraction:.0%} overlap require a "
            f"{stride_seconds_float:g}-second stride, which cannot "
            f"be represented exactly using "
            f"{source_window_seconds}-second source blocks."
        )

    source_blocks_per_stride = int(
        stride_block_ratio
    )

    if source_blocks_per_stride <= 0:
        raise ValueError(
            "The source-block stride must be positive."
        )

    if (
        recording_base_windows
        < source_blocks_per_window
    ):
        start_indices: tuple[int, ...] = ()
    else:
        start_indices = tuple(
            range(
                0,
                (
                    recording_base_windows
                    - source_blocks_per_window
                    + 1
                ),
                source_blocks_per_stride,
            )
        )

    return LongWindowPlan(
        duration_seconds=duration_seconds,
        stride_seconds=int(
            round(
                stride_seconds_float
            )
        ),
        source_blocks_per_window=(
            source_blocks_per_window
        ),
        source_blocks_per_stride=(
            source_blocks_per_stride
        ),
        start_indices=start_indices,
    )


def concatenate_signal_blocks(
    array: np.ndarray,
    *,
    start_index: int,
    block_count: int,
) -> np.ndarray:
    """Concatenate consecutive source windows along time."""

    source = np.asarray(array)

    if source.ndim not in {
        2,
        3,
    }:
        raise ValueError(
            "Source array must have shape "
            "(windows, samples) or "
            "(windows, samples, channels)."
        )

    end_index = (
        start_index
        + block_count
    )

    if (
        start_index < 0
        or end_index > source.shape[0]
    ):
        raise IndexError(
            "Requested source-block interval is out of range."
        )

    selected = source[
        start_index:end_index
    ]

    if source.ndim == 2:
        return selected.reshape(
            -1
        )

    return selected.reshape(
        -1,
        source.shape[2],
    )


def model_feature_columns(
    frame: pd.DataFrame,
) -> list[str]:
    """Return feature-only columns."""

    return [
        column
        for column in frame.columns
        if column not in METADATA_COLUMNS
    ]


def validate_long_feature_table(
    frame: pd.DataFrame,
    name: str,
) -> None:
    """Validate identifiers and finite numerical feature values."""

    missing_metadata = [
        column
        for column in METADATA_COLUMNS
        if column not in frame.columns
    ]

    if missing_metadata:
        raise RuntimeError(
            f"{name} lacks metadata columns: "
            f"{missing_metadata}"
        )

    if frame.empty:
        raise RuntimeError(
            f"{name} is empty."
        )

    if frame[
        "segment_id"
    ].duplicated().any():
        raise RuntimeError(
            f"{name} contains duplicate segment identifiers."
        )

    features = model_feature_columns(
        frame
    )

    if not features:
        raise RuntimeError(
            f"{name} contains no model features."
        )

    values = frame[
        features
    ].to_numpy(dtype=float)

    if not np.isfinite(
        values
    ).all():
        count = int(
            (
                ~np.isfinite(values)
            ).sum()
        )

        raise RuntimeError(
            f"{name} contains {count} non-finite values."
        )


def fuse_long_tables(
    left: pd.DataFrame,
    right: pd.DataFrame,
) -> pd.DataFrame:
    """Fuse two long-window modality tables by segment ID."""

    left_features = model_feature_columns(
        left
    )

    right_features = model_feature_columns(
        right
    )

    collisions = set(
        left_features
    ).intersection(
        right_features
    )

    if collisions:
        raise RuntimeError(
            "Feature-name collision during fusion: "
            f"{sorted(collisions)[:10]}"
        )

    fused = left[
        METADATA_COLUMNS
        + left_features
    ].merge(
        right[
            [
                "segment_id",
            ]
            + right_features
        ],
        on="segment_id",
        how="inner",
        validate="one_to_one",
    )

    return fused


def duration_directory_name(
    duration_seconds: int,
    overlap_fraction: float,
) -> str:
    """Return a stable output directory name."""

    overlap_percent = int(
        round(
            overlap_fraction
            * 100
        )
    )

    return (
        f"window_{duration_seconds:02d}s_"
        f"overlap_{overlap_percent:02d}pct"
    )


def recording_metadata(
    recording_manifest: pd.DataFrame,
) -> tuple[str, int, int]:
    """Resolve invariant participant, phase, and label metadata."""

    participants = (
        recording_manifest[
            "participant"
        ]
        .astype(str)
        .unique()
    )

    phases = (
        recording_manifest[
            "phase"
        ]
        .astype(int)
        .unique()
    )

    labels = (
        recording_manifest[
            "label"
        ]
        .astype(int)
        .unique()
    )

    if (
        len(participants) != 1
        or len(phases) != 1
        or len(labels) != 1
    ):
        raise RuntimeError(
            "A recording manifest contains inconsistent metadata."
        )

    return (
        str(participants[0]),
        int(phases[0]),
        int(labels[0]),
    )


def extract_duration_tables(
    *,
    duration_seconds: int,
    overlap_fraction: float,
    source_window_seconds: int,
    source_directory: Path,
    manifest: pd.DataFrame,
    eeg_channels: list[str],
    eeg_bands: dict[str, list[float]],
    pupil_bands: dict[str, list[float]],
    nperseg: int,
) -> tuple[
    dict[str, pd.DataFrame],
    pd.DataFrame,
]:
    """Extract all seven modality tables for one duration."""

    eeg_rows: list[dict[str, Any]] = []
    ecg_rows: list[dict[str, Any]] = []
    pupil_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []

    grouped = manifest.groupby(
        "recording_archive",
        sort=True,
    )

    recording_total = grouped.ngroups

    for recording_number, (
        relative_archive,
        recording_manifest,
    ) in enumerate(
        grouped,
        start=1,
    ):
        recording_manifest = (
            recording_manifest
            .sort_values(
                "archive_window_index"
            )
            .reset_index(drop=True)
        )

        participant, phase, label = (
            recording_metadata(
                recording_manifest
            )
        )

        archive_path = (
            source_directory
            / str(relative_archive)
        )

        if not archive_path.is_file():
            raise FileNotFoundError(
                f"Corrected archive not found: {archive_path}"
            )

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

            pupil = np.asarray(
                archive["pupil"],
                dtype=float,
            )

            pupil_available_source = np.asarray(
                archive["pupil_available"],
                dtype=bool,
            )

            archive_channels = [
                str(value)
                for value in archive[
                    "eeg_channels"
                ].tolist()
            ]

            if archive_channels != eeg_channels:
                raise RuntimeError(
                    f"EEG channel mismatch in {archive_path}: "
                    f"{archive_channels}"
                )

            if not (
                len(eeg)
                == len(ecg)
                == len(pupil)
                == len(pupil_available_source)
                == len(recording_manifest)
            ):
                raise RuntimeError(
                    f"Source-window count mismatch in {archive_path}."
                )

            expected_archive_indices = np.arange(
                len(recording_manifest),
                dtype=int,
            )

            observed_archive_indices = (
                recording_manifest[
                    "archive_window_index"
                ].to_numpy(dtype=int)
            )

            if not np.array_equal(
                expected_archive_indices,
                observed_archive_indices,
            ):
                raise RuntimeError(
                    f"Non-contiguous archive indices in {archive_path}."
                )

            eeg_sampling_rate = float(
                archive[
                    "eeg_ecg_sampling_rate_hz"
                ]
            )

            pupil_sampling_rate = float(
                archive[
                    "pupil_sampling_rate_hz"
                ]
            )

            recorded_source_seconds = float(
                archive[
                    "window_seconds"
                ]
            )

            if not np.isclose(
                recorded_source_seconds,
                source_window_seconds,
                atol=1e-9,
                rtol=1e-9,
            ):
                raise RuntimeError(
                    f"Expected {source_window_seconds}-second source "
                    f"windows but found {recorded_source_seconds} "
                    f"in {archive_path}."
                )

            plan = build_long_window_plan(
                len(eeg),
                source_window_seconds=(
                    source_window_seconds
                ),
                duration_seconds=(
                    duration_seconds
                ),
                overlap_fraction=(
                    overlap_fraction
                ),
            )

            for long_window_index, start in enumerate(
                plan.start_indices
            ):
                end = (
                    start
                    + plan.source_blocks_per_window
                )

                segment_id = (
                    f"{participant}_phase{phase}_"
                    f"{duration_seconds:02d}s_"
                    f"start{start:03d}"
                )

                pupil_available = bool(
                    np.all(
                        pupil_available_source[
                            start:end
                        ]
                    )
                )

                metadata: dict[str, Any] = {
                    "segment_id":
                        segment_id,
                    "participant":
                        participant,
                    "phase":
                        phase,
                    "label":
                        label,
                    "long_window_index":
                        long_window_index,
                    "start_base_window":
                        start,
                    "end_base_window_exclusive":
                        end,
                    "start_seconds":
                        float(
                            start
                            * source_window_seconds
                        ),
                    "end_seconds":
                        float(
                            start
                            * source_window_seconds
                            + duration_seconds
                        ),
                    "window_seconds":
                        duration_seconds,
                    "stride_seconds":
                        plan.stride_seconds,
                    "overlap_fraction":
                        overlap_fraction,
                    "pupil_available":
                        pupil_available,
                }

                long_eeg = concatenate_signal_blocks(
                    eeg,
                    start_index=start,
                    block_count=(
                        plan.source_blocks_per_window
                    ),
                )

                long_ecg = concatenate_signal_blocks(
                    ecg,
                    start_index=start,
                    block_count=(
                        plan.source_blocks_per_window
                    ),
                )

                eeg_features = extract_eeg_features(
                    long_eeg,
                    eeg_sampling_rate,
                    eeg_channels,
                    eeg_bands,
                    nperseg,
                )

                ecg_features = extract_ecg_features(
                    long_ecg,
                    eeg_sampling_rate,
                    nperseg,
                )

                eeg_rows.append(
                    {
                        **metadata,
                        **eeg_features,
                    }
                )

                ecg_rows.append(
                    {
                        **metadata,
                        **ecg_features,
                    }
                )

                if pupil_available:
                    long_pupil = (
                        concatenate_signal_blocks(
                            pupil,
                            start_index=start,
                            block_count=(
                                plan.source_blocks_per_window
                            ),
                        )
                    )

                    if not np.isfinite(
                        long_pupil
                    ).all():
                        raise RuntimeError(
                            f"{segment_id}: a pupil-available long "
                            f"window contains non-finite values."
                        )

                    pupil_features = (
                        extract_pupil_features(
                            long_pupil,
                            pupil_sampling_rate,
                            pupil_bands,
                            nperseg,
                        )
                    )

                    pupil_rows.append(
                        {
                            **metadata,
                            **pupil_features,
                        }
                    )

                manifest_rows.append(
                    metadata
                )

        if (
            recording_number % 5 == 0
            or recording_number
            == recording_total
        ):
            print(
                f"{duration_seconds:02d}s: processed recordings "
                f"{recording_number}/{recording_total}",
                flush=True,
            )

    eeg_table = pd.DataFrame(
        eeg_rows
    )

    ecg_table = pd.DataFrame(
        ecg_rows
    )

    pupil_table = pd.DataFrame(
        pupil_rows
    )

    validate_long_feature_table(
        eeg_table,
        "EEG",
    )

    validate_long_feature_table(
        ecg_table,
        "ECG",
    )

    validate_long_feature_table(
        pupil_table,
        "Pupil",
    )

    ecg_eeg = fuse_long_tables(
        ecg_table,
        eeg_table,
    )

    ecg_pupil = fuse_long_tables(
        ecg_table,
        pupil_table,
    )

    eeg_pupil = fuse_long_tables(
        eeg_table,
        pupil_table,
    )

    trimodal = fuse_long_tables(
        ecg_eeg,
        pupil_table,
    )

    tables = {
        "ECG": ecg_table,
        "EEG": eeg_table,
        "Pupil": pupil_table,
        "ECG_EEG": ecg_eeg,
        "ECG_Pupil": ecg_pupil,
        "EEG_Pupil": eeg_pupil,
        "ECG_EEG_Pupil": trimodal,
    }

    for name, table in tables.items():
        validate_long_feature_table(
            table,
            name,
        )

    long_manifest = pd.DataFrame(
        manifest_rows
    )

    if long_manifest[
        "segment_id"
    ].duplicated().any():
        raise RuntimeError(
            "Long-window manifest contains duplicate segment IDs."
        )

    if len(eeg_table) != len(
        long_manifest
    ):
        raise RuntimeError(
            "EEG rows do not match the long-window manifest."
        )

    if len(ecg_table) != len(
        long_manifest
    ):
        raise RuntimeError(
            "ECG rows do not match the long-window manifest."
        )

    expected_pupil_rows = int(
        long_manifest[
            "pupil_available"
        ].sum()
    )

    if len(pupil_table) != expected_pupil_rows:
        raise RuntimeError(
            "Pupil rows do not match the conservative "
            "long-window availability policy."
        )

    return tables, long_manifest


def build_long_window_features(
    config: dict[str, Any],
    *,
    write_outputs: bool,
) -> dict[str, Any]:
    """Build the configured longer-window feature collections."""

    source_directory = resolve_project_path(
        config[
            "datasets"
        ][
            "internal_xr"
        ][
            "output_dir"
        ]
    )

    source_manifest_path = (
        source_directory
        / "window_manifest.csv"
    )

    if not source_manifest_path.is_file():
        raise FileNotFoundError(
            f"Corrected source manifest not found: "
            f"{source_manifest_path}"
        )

    feature_config = config[
        "features"
    ]

    long_config = feature_config[
        "long_windows"
    ]

    source_window_seconds = int(
        long_config[
            "source_window_seconds"
        ]
    )

    durations = [
        int(value)
        for value in long_config[
            "durations_seconds"
        ]
    ]

    overlap_fraction = float(
        long_config[
            "overlap_fraction"
        ]
    )

    output_directory = resolve_project_path(
        long_config[
            "output_dir"
        ]
    )

    manifest = pd.read_csv(
        source_manifest_path,
        low_memory=False,
    )

    required_manifest_columns = {
        "recording_archive",
        "archive_window_index",
        "participant",
        "phase",
        "label",
    }

    missing_columns = (
        required_manifest_columns
        - set(manifest.columns)
    )

    if missing_columns:
        raise RuntimeError(
            "Corrected source manifest is missing columns: "
            f"{sorted(missing_columns)}"
        )

    eeg_channels = list(
        config[
            "eeg"
        ][
            "internal_channels"
        ]
    )

    eeg_bands = feature_config[
        "eeg_bands_hz"
    ]

    pupil_bands = feature_config[
        "pupil_bands_hz"
    ]

    nperseg = int(
        feature_config[
            "welch_nperseg"
        ]
    )

    duration_outputs: dict[
        int,
        tuple[
            dict[str, pd.DataFrame],
            pd.DataFrame,
        ],
    ] = {}

    summary: dict[str, Any] = {
        "dataset":
            "internal_xr_attention",
        "source_directory":
            str(source_directory),
        "source_manifest":
            str(source_manifest_path),
        "source_window_seconds":
            source_window_seconds,
        "durations_seconds":
            durations,
        "overlap_fraction":
            overlap_fraction,
        "pupil_availability_policy":
            (
                "all constituent corrected four-second "
                "windows must be pupil-available"
            ),
        "normalization_applied":
            False,
        "feature_selection_applied":
            False,
        "participant_splitting_applied":
            False,
        "durations": {},
    }

    for duration in durations:
        print(
            f"\n===== BUILD {duration}-SECOND WINDOWS =====",
            flush=True,
        )

        tables, long_manifest = (
            extract_duration_tables(
                duration_seconds=duration,
                overlap_fraction=(
                    overlap_fraction
                ),
                source_window_seconds=(
                    source_window_seconds
                ),
                source_directory=(
                    source_directory
                ),
                manifest=manifest,
                eeg_channels=eeg_channels,
                eeg_bands=eeg_bands,
                pupil_bands=pupil_bands,
                nperseg=nperseg,
            )
        )

        duration_outputs[
            duration
        ] = (
            tables,
            long_manifest,
        )

        summary[
            "durations"
        ][
            str(duration)
        ] = {
            "directory":
                duration_directory_name(
                    duration,
                    overlap_fraction,
                ),
            "total_windows":
                int(
                    len(
                        long_manifest
                    )
                ),
            "participants":
                int(
                    long_manifest[
                        "participant"
                    ].nunique()
                ),
            "pupil_available_windows":
                int(
                    long_manifest[
                        "pupil_available"
                    ].sum()
                ),
            "label_counts": {
                str(int(label)): int(count)
                for label, count
                in (
                    long_manifest[
                        "label"
                    ]
                    .value_counts()
                    .sort_index()
                    .items()
                )
            },
            "phase_counts": {
                str(int(phase)): int(count)
                for phase, count
                in (
                    long_manifest[
                        "phase"
                    ]
                    .value_counts()
                    .sort_index()
                    .items()
                )
            },
            "tables": {
                name: {
                    "rows":
                        int(len(table)),
                    "features":
                        int(
                            len(
                                model_feature_columns(
                                    table
                                )
                            )
                        ),
                }
                for name, table
                in tables.items()
            },
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
                "Refusing to overwrite existing long-window output: "
                f"{output_directory}"
            )

        if temporary_directory.exists():
            raise FileExistsError(
                "Temporary long-window output already exists: "
                f"{temporary_directory}"
            )

        temporary_directory.mkdir(
            parents=True,
            exist_ok=False,
        )

        for duration, (
            tables,
            long_manifest,
        ) in duration_outputs.items():
            duration_directory = (
                temporary_directory
                / duration_directory_name(
                    duration,
                    overlap_fraction,
                )
            )

            duration_directory.mkdir(
                parents=True,
                exist_ok=False,
            )

            long_manifest.to_csv(
                duration_directory
                / "long_window_manifest.csv",
                index=False,
            )

            for table_name, table in tables.items():
                table.to_csv(
                    duration_directory
                    / f"{table_name}_features.csv",
                    index=False,
                )

            (
                duration_directory
                / "feature_schema.json"
            ).write_text(
                json.dumps(
                    {
                        table_name:
                            model_feature_columns(
                                table
                            )
                        for table_name, table
                        in tables.items()
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )

        (
            temporary_directory
            / "long_window_summary.json"
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
        "\n===== LONG-WINDOW FEATURE SUMMARY ====="
    )

    for duration in durations:
        duration_summary = (
            summary[
                "durations"
            ][
                str(duration)
            ]
        )

        print(
            f"{duration:2d}s | "
            f"windows={duration_summary['total_windows']:4d} | "
            f"pupil={duration_summary['pupil_available_windows']:4d} | "
            f"labels={duration_summary['label_counts']}"
        )

        for table_name, table_summary in (
            duration_summary[
                "tables"
            ].items()
        ):
            print(
                f"      {table_name:18s} "
                f"rows={table_summary['rows']:4d}, "
                f"features={table_summary['features']:3d}"
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
            "Build exact 50%-overlapping longer-window "
            "internal XR feature matrices."
        )
    )

    parser.add_argument(
        "--write",
        action="store_true",
        help="Write all longer-window feature artifacts.",
    )

    arguments = parser.parse_args()

    config = load_revision_config(
        check_input_paths=True
    )

    build_long_window_features(
        config,
        write_outputs=arguments.write,
    )


if __name__ == "__main__":
    main()