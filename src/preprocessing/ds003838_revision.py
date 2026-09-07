"""Leakage-safe event-locked preprocessing for OpenNeuro ds003838."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable

import mne
import numpy as np
import pandas as pd
from scipy.signal import (
    butter,
    filtfilt,
    iirnotch,
    resample_poly,
    sosfiltfilt,
)

from src.core.config import load_revision_config
from src.data_loaders.ds003838_manifest import (
    _align_trials,
    _build_trials,
    _clock_segments,
    _map_eeg_time,
    _prepare_events,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path.resolve()


def rational_resampling_ratio(
    original_rate_hz: float,
    target_rate_hz: float,
) -> tuple[int, int]:
    """Return a stable rational approximation for polyphase resampling."""

    if original_rate_hz <= 0 or target_rate_hz <= 0:
        raise ValueError("Sampling rates must be positive.")

    ratio = Fraction(
        float(target_rate_hz) / float(original_rate_hz)
    ).limit_denominator(10000)

    return int(ratio.numerator), int(ratio.denominator)


def filter_notch_and_resample(
    signal: np.ndarray,
    *,
    original_rate_hz: float,
    target_rate_hz: float,
    lowcut_hz: float,
    highcut_hz: float,
    filter_order: int,
    notch_frequency_hz: float,
    notch_quality_factor: float,
) -> np.ndarray:
    """Notch, bandpass, and polyphase-resample a signal array."""

    array = np.asarray(signal, dtype=float)

    if array.ndim == 1:
        array = array[:, np.newaxis]

    if array.ndim != 2:
        raise ValueError(
            "Signal must be samples by channels."
        )

    if not np.isfinite(array).all():
        raise ValueError(
            "Signal contains nonfinite values."
        )

    nyquist = float(original_rate_hz) / 2.0

    if not 0.0 < lowcut_hz < highcut_hz < nyquist:
        raise ValueError(
            "Bandpass limits are incompatible with sampling rate."
        )

    processed = array

    if 0.0 < notch_frequency_hz < nyquist:
        numerator, denominator = iirnotch(
            w0=float(notch_frequency_hz),
            Q=float(notch_quality_factor),
            fs=float(original_rate_hz),
        )

        processed = filtfilt(
            numerator,
            denominator,
            processed,
            axis=0,
        )

    sos = butter(
        int(filter_order),
        [float(lowcut_hz), float(highcut_hz)],
        btype="bandpass",
        fs=float(original_rate_hz),
        output="sos",
    )

    processed = sosfiltfilt(
        sos,
        processed,
        axis=0,
    )

    up, down = rational_resampling_ratio(
        original_rate_hz,
        target_rate_hz,
    )

    processed = resample_poly(
        processed,
        up=up,
        down=down,
        axis=0,
    )

    if not np.isfinite(processed).all():
        raise RuntimeError(
            "Signal preprocessing produced nonfinite values."
        )

    return processed


def trial_anchor_time(
    trial: dict[str, Any],
    anchor: str,
) -> float:
    """Resolve a validated temporal anchor for one digit-span trial."""

    normalized = str(anchor).strip().lower()

    if normalized == "final_digit_offset":
        return float(
            trial["final_offset"]
        )

    if normalized == "first_digit_onset":
        events = trial.get(
            "events",
            []
        )

        if not events:
            raise ValueError(
                "Trial has no events for first-digit anchoring."
            )

        return float(
            events[0]["time"]
        )

    raise ValueError(
        f"Unsupported trial-window anchor: {anchor}"
    )


def extract_padded_signal_window(
    raw: mne.io.BaseRaw,
    *,
    picks: Iterable[str],
    window_start_seconds: float,
    window_stop_seconds: float,
    target_rate_hz: float,
    lowcut_hz: float,
    highcut_hz: float,
    filter_order: int,
    notch_frequency_hz: float,
    notch_quality_factor: float,
    padding_seconds: float,
) -> np.ndarray:
    """Extract one event window with temporal padding for filtering."""

    start = float(window_start_seconds)
    stop = float(window_stop_seconds)

    if stop <= start:
        raise ValueError(
            "Window stop must be after window start."
        )

    recording_stop = float(raw.times[-1])

    if start < 0.0 or stop > recording_stop:
        raise ValueError(
            "Requested window is outside recording coverage."
        )

    padded_start = max(
        0.0,
        start - float(padding_seconds),
    )

    padded_stop = min(
        recording_stop,
        stop + float(padding_seconds),
    )

    indices = raw.time_as_index(
        [padded_start, padded_stop],
        use_rounding=True,
    )

    start_index = int(indices[0])
    stop_index = int(indices[1])

    if stop_index <= start_index:
        raise RuntimeError(
            "Invalid padded sample interval."
        )

    signal = raw.get_data(
        picks=list(picks),
        start=start_index,
        stop=stop_index,
    ).T

    processed = filter_notch_and_resample(
        signal,
        original_rate_hz=float(raw.info["sfreq"]),
        target_rate_hz=float(target_rate_hz),
        lowcut_hz=float(lowcut_hz),
        highcut_hz=float(highcut_hz),
        filter_order=int(filter_order),
        notch_frequency_hz=float(notch_frequency_hz),
        notch_quality_factor=float(notch_quality_factor),
    )

    crop_start = int(
        round(
            (start - padded_start)
            * float(target_rate_hz)
        )
    )

    expected_samples = int(
        round(
            (stop - start)
            * float(target_rate_hz)
        )
    )

    crop_stop = crop_start + expected_samples
    cropped = processed[crop_start:crop_stop]

    if cropped.shape[0] != expected_samples:
        raise RuntimeError(
            "Resampled event window has an unexpected length: "
            f"expected {expected_samples}, observed {cropped.shape[0]}"
        )

    return cropped



def prepare_pupil_frame(
    path: str | Path,
    *,
    timestamp_field: str,
    gaze_timestamp_field: str,
    gaze_x_field: str,
    gaze_y_field: str,
    diameter_field: str,
) -> dict[str, pd.DataFrame]:
    """Load separate pupil-diameter and gaze streams.

    The ds003838 eye-tracking TSV vertically interleaves pupil-detection
    rows and gaze-estimation rows. No row is expected to contain both
    pupil and gaze measurements.
    """

    source = Path(path)

    header = pd.read_csv(
        source,
        sep="\t",
        nrows=0,
    )

    required = {
        timestamp_field,
        gaze_timestamp_field,
        gaze_x_field,
        gaze_y_field,
        diameter_field,
        "confidence",
    }

    missing = required - set(header.columns)

    if missing:
        raise ValueError(
            f"Eye-tracking file lacks required fields: "
            f"{sorted(missing)}"
        )

    optional = {
        "blink",
        "eye_id",
    }

    use_columns = sorted(
        required
        | (
            optional
            & set(header.columns)
        )
    )

    pupil_parts: list[pd.DataFrame] = []
    gaze_parts: list[pd.DataFrame] = []

    for chunk in pd.read_csv(
        source,
        sep="\t",
        usecols=use_columns,
        low_memory=False,
        chunksize=500000,
    ):
        for column in use_columns:
            chunk[column] = pd.to_numeric(
                chunk[column],
                errors="coerce",
            )

        pupil_mask = (
            np.isfinite(chunk[timestamp_field])
            & np.isfinite(chunk[diameter_field])
        )

        gaze_mask = (
            np.isfinite(chunk[gaze_timestamp_field])
            & np.isfinite(chunk[gaze_x_field])
            & np.isfinite(chunk[gaze_y_field])
        )

        if pupil_mask.any():
            pupil_columns = [
                timestamp_field,
                diameter_field,
                "confidence",
            ]

            for optional_column in [
                "blink",
                "eye_id",
            ]:
                if optional_column in chunk.columns:
                    pupil_columns.append(optional_column)

            pupil_parts.append(
                chunk.loc[
                    pupil_mask,
                    pupil_columns,
                ].copy()
            )

        if gaze_mask.any():
            gaze_columns = [
                gaze_timestamp_field,
                gaze_x_field,
                gaze_y_field,
                "confidence",
            ]

            if "blink" in chunk.columns:
                gaze_columns.append("blink")

            gaze_parts.append(
                chunk.loc[
                    gaze_mask,
                    gaze_columns,
                ].copy()
            )

    if not pupil_parts:
        raise RuntimeError(
            "No pupil-diameter stream could be reconstructed."
        )

    if not gaze_parts:
        raise RuntimeError(
            "No gaze stream could be reconstructed."
        )

    pupil_stream = pd.concat(
        pupil_parts,
        ignore_index=True,
    ).sort_values(
        timestamp_field,
        kind="stable",
    ).reset_index(drop=True)

    gaze_stream = pd.concat(
        gaze_parts,
        ignore_index=True,
    ).sort_values(
        gaze_timestamp_field,
        kind="stable",
    ).reset_index(drop=True)

    return {
        "pupil": pupil_stream,
        "gaze": gaze_stream,
    }


def _slice_time_window(
    frame: pd.DataFrame,
    *,
    timestamp_field: str,
    start_seconds: float,
    stop_seconds: float,
) -> pd.DataFrame:
    """Return a time interval using binary search on sorted timestamps."""

    timestamps = frame[
        timestamp_field
    ].to_numpy(dtype=float)

    left = int(
        np.searchsorted(
            timestamps,
            float(start_seconds),
            side="left",
        )
    )

    right = int(
        np.searchsorted(
            timestamps,
            float(stop_seconds),
            side="left",
        )
    )

    return frame.iloc[left:right]


def _interpolate_tracking_stream(
    frame: pd.DataFrame,
    *,
    timestamp_field: str,
    value_fields: list[str],
    grid: np.ndarray,
    start_seconds: float,
    stop_seconds: float,
    minimum_confidence: float,
    minimum_valid_fraction: float,
    maximum_gap_seconds: float,
    require_positive_values: bool,
) -> tuple[np.ndarray | None, dict[str, float]]:
    """Validate and interpolate one independently sampled stream."""

    selected = _slice_time_window(
        frame,
        timestamp_field=timestamp_field,
        start_seconds=start_seconds,
        stop_seconds=stop_seconds,
    ).copy()

    candidate_count = int(len(selected))

    empty_quality = {
        "candidate_samples": candidate_count,
        "valid_samples": 0,
        "valid_fraction": 0.0,
        "maximum_gap_seconds": float("inf"),
    }

    if candidate_count < 3:
        return None, empty_quality

    valid = np.isfinite(
        selected[timestamp_field].to_numpy(dtype=float)
    )

    confidence = selected[
        "confidence"
    ].to_numpy(dtype=float)

    valid &= (
        np.isfinite(confidence)
        & (
            confidence
            >= float(minimum_confidence)
        )
    )

    if "blink" in selected.columns:
        blink = (
            selected["blink"]
            .fillna(0.0)
            .to_numpy(dtype=float)
        )

        valid &= (
            np.isfinite(blink)
            & (blink <= 0.0)
        )

    for field in value_fields:
        values = selected[field].to_numpy(dtype=float)

        valid &= np.isfinite(values)

        if require_positive_values:
            valid &= values > 0.0

    valid_frame = selected.loc[
        valid,
        [
            timestamp_field,
            *value_fields,
        ],
    ].copy()

    valid_count = int(len(valid_frame))

    valid_fraction = (
        valid_count / candidate_count
        if candidate_count
        else 0.0
    )

    if (
        valid_count < 3
        or valid_fraction
        < float(minimum_valid_fraction)
    ):
        return None, {
            "candidate_samples": candidate_count,
            "valid_samples": valid_count,
            "valid_fraction": float(valid_fraction),
            "maximum_gap_seconds": float("inf"),
        }

    grouped = (
        valid_frame.groupby(
            timestamp_field,
            as_index=False,
            sort=True,
        )
        .median(numeric_only=True)
    )

    times = grouped[
        timestamp_field
    ].to_numpy(dtype=float)

    differences = np.diff(times)

    maximum_gap = (
        float(np.max(differences))
        if len(differences)
        else float("inf")
    )

    boundary_tolerance = float(
        maximum_gap_seconds
    )

    coverage_passed = (
        times[0]
        <= float(start_seconds)
        + boundary_tolerance
        and times[-1]
        >= float(stop_seconds)
        - boundary_tolerance
        and maximum_gap
        <= float(maximum_gap_seconds)
    )

    quality = {
        "candidate_samples": candidate_count,
        "valid_samples": valid_count,
        "valid_fraction": float(valid_fraction),
        "maximum_gap_seconds": maximum_gap,
    }

    if not coverage_passed:
        return None, quality

    interpolated_columns = []

    for field in value_fields:
        interpolated_columns.append(
            np.interp(
                grid,
                times,
                grouped[field].to_numpy(dtype=float),
            )
        )

    interpolated = np.column_stack(
        interpolated_columns
    )

    if not np.isfinite(interpolated).all():
        raise RuntimeError(
            "Tracking-stream interpolation produced "
            "nonfinite values."
        )

    return interpolated, quality


def interpolate_pupil_window(
    streams: dict[str, pd.DataFrame],
    *,
    window_start_seconds: float,
    window_stop_seconds: float,
    timestamp_field: str,
    gaze_timestamp_field: str,
    gaze_x_field: str,
    gaze_y_field: str,
    diameter_field: str,
    target_rate_hz: float,
    minimum_confidence: float,
    minimum_valid_fraction: float,
    maximum_gap_seconds: float,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Merge independent gaze and pupil streams on a uniform grid."""

    start = float(window_start_seconds)
    stop = float(window_stop_seconds)

    if stop <= start:
        raise ValueError(
            "Window stop must be after window start."
        )

    expected_samples = int(
        round(
            (stop - start)
            * float(target_rate_hz)
        )
    )

    grid = (
        start
        + (
            np.arange(
                expected_samples,
                dtype=float,
            )
            / float(target_rate_hz)
        )
    )

    pupil_values, pupil_quality = (
        _interpolate_tracking_stream(
            streams["pupil"],
            timestamp_field=timestamp_field,
            value_fields=[diameter_field],
            grid=grid,
            start_seconds=start,
            stop_seconds=stop,
            minimum_confidence=minimum_confidence,
            minimum_valid_fraction=minimum_valid_fraction,
            maximum_gap_seconds=maximum_gap_seconds,
            require_positive_values=True,
        )
    )

    gaze_values, gaze_quality = (
        _interpolate_tracking_stream(
            streams["gaze"],
            timestamp_field=gaze_timestamp_field,
            value_fields=[
                gaze_x_field,
                gaze_y_field,
            ],
            grid=grid,
            start_seconds=start,
            stop_seconds=stop,
            minimum_confidence=minimum_confidence,
            minimum_valid_fraction=minimum_valid_fraction,
            maximum_gap_seconds=maximum_gap_seconds,
            require_positive_values=False,
        )
    )

    available = (
        pupil_values is not None
        and gaze_values is not None
    )

    quality: dict[str, Any] = {
        "pupil_candidate_samples":
            pupil_quality["candidate_samples"],
        "pupil_valid_samples":
            pupil_quality["valid_samples"],
        "pupil_valid_fraction":
            pupil_quality["valid_fraction"],
        "pupil_maximum_gap_seconds":
            pupil_quality["maximum_gap_seconds"],
        "gaze_candidate_samples":
            gaze_quality["candidate_samples"],
        "gaze_valid_samples":
            gaze_quality["valid_samples"],
        "gaze_valid_fraction":
            gaze_quality["valid_fraction"],
        "gaze_maximum_gap_seconds":
            gaze_quality["maximum_gap_seconds"],
        "pupil_available": bool(available),
    }

    if not available:
        return None, quality

    radius = pupil_values[:, 0] / 2.0

    window = np.column_stack(
        [
            gaze_values[:, 0],
            gaze_values[:, 1],
            radius,
        ]
    )

    if (
        window.shape != (expected_samples, 3)
        or not np.isfinite(window).all()
    ):
        raise RuntimeError(
            "Merged gazepupil window is invalid."
        )

    return window, quality

def reconstruct_clock_segments(
    eeg_trials: list[dict[str, Any]],
    pupil_trials: list[dict[str, Any]],
    alignment: list[tuple[int, int]],
    *,
    jump_threshold_seconds: float,
) -> list[dict[str, Any]]:
    """Reconstruct the scanner-validated EEG-to-pupil clock map."""

    eeg_times: list[float] = []
    pupil_times: list[float] = []

    for eeg_index, pupil_index in alignment:
        eeg_events = eeg_trials[eeg_index]["events"]
        pupil_events = pupil_trials[pupil_index]["events"]

        if len(eeg_events) != len(pupil_events):
            raise RuntimeError(
                "Aligned trials contain different event counts."
            )

        for eeg_event, pupil_event in zip(
            eeg_events,
            pupil_events,
        ):
            eeg_times.append(float(eeg_event["time"]))
            pupil_times.append(float(pupil_event["time"]))

    segments, _ = _clock_segments(
        np.asarray(eeg_times, dtype=float),
        np.asarray(pupil_times, dtype=float),
        jump_threshold_seconds=float(
            jump_threshold_seconds
        ),
    )

    return segments


def preprocess_subject(
    manifest_row: pd.Series,
    config: dict[str, Any],
) -> tuple[dict[str, np.ndarray], pd.DataFrame, dict[str, Any]]:
    """Preprocess all usable memory trials for one ds003838 subject."""

    subject = str(manifest_row["subject"])
    dataset_config = config["datasets"]["ds003838"]
    preprocessing = dataset_config["preprocessing"]
    synchronization = dataset_config["synchronization"]
    window_config = dataset_config["window"]

    channels = list(
        config["eeg"]["ds003838_common_channels"]
    )

    label_map = {
        int(key): int(value)
        for key, value in dataset_config["label_map"].items()
    }

    eeg_events = _prepare_events(
        Path(str(manifest_row["eeg_events_path"])),
        code_column="value",
        time_column="onset",
    )

    pupil_events = _prepare_events(
        Path(str(manifest_row["pupil_events_path"])),
        code_column="label",
        time_column="timestamp",
    )

    eeg_trials = _build_trials(eeg_events)
    pupil_trials = _build_trials(pupil_events)
    alignment = _align_trials(
        eeg_trials,
        pupil_trials,
    )

    segments = reconstruct_clock_segments(
        eeg_trials,
        pupil_trials,
        alignment,
        jump_threshold_seconds=float(
            synchronization[
                "clock_jump_threshold_seconds"
            ]
        ),
    )

    eeg_raw = mne.io.read_raw_eeglab(
        Path(str(manifest_row["eeg_path"])),
        preload=False,
        verbose="ERROR",
    )

    ecg_raw = mne.io.read_raw_eeglab(
        Path(str(manifest_row["ecg_path"])),
        preload=False,
        verbose="ERROR",
    )

    missing_channels = [
        channel
        for channel in channels
        if channel not in eeg_raw.ch_names
    ]

    if missing_channels:
        raise RuntimeError(
            f"{subject} lacks required EEG channels: "
            f"{missing_channels}"
        )

    if "ECG" not in ecg_raw.ch_names:
        raise RuntimeError(
            f"{subject} lacks the ECG channel."
        )

    timestamp_field = "pupil_timestamp"
    gaze_timestamp_field = "gaze_timestamp"
    gaze_x_field = str(
        preprocessing["pupil_gaze_x_field"]
    )
    gaze_y_field = str(
        preprocessing["pupil_gaze_y_field"]
    )
    diameter_field = str(
        preprocessing["pupil_diameter_field"]
    )

    pupil_streams = prepare_pupil_frame(
        manifest_row["pupil_path"],
        timestamp_field=timestamp_field,
        gaze_timestamp_field=gaze_timestamp_field,
        gaze_x_field=gaze_x_field,
        gaze_y_field=gaze_y_field,
        diameter_field=diameter_field,
    )

    duration_seconds = float(
        window_config["duration_seconds"]
    )

    delay_seconds = float(
        window_config["start_offset_seconds"]
    )

    window_anchor = str(
        window_config.get(
            "anchor",
            "final_digit_offset",
        )
    ).strip().lower()

    if window_anchor not in {
        "final_digit_offset",
        "first_digit_onset",
    }:
        raise ValueError(
            f"Unsupported ds003838 window anchor: "
            f"{window_anchor}"
        )


    pupil_target_rate_hz = float(
        preprocessing[
            "pupil_target_sampling_rate_hz"
        ]
    )

    expected_pupil_samples = int(
        round(
            duration_seconds
            * pupil_target_rate_hz
        )
    )

    eeg_windows: list[np.ndarray] = []
    ecg_windows: list[np.ndarray] = []
    pupil_windows: list[np.ndarray] = []
    metadata_rows: list[dict[str, Any]] = []
    pupil_quality_rows: list[dict[str, Any]] = []

    memory_counter = 0

    for eeg_index, pupil_index in alignment:
        eeg_trial = eeg_trials[eeg_index]

        if eeg_trial["condition"] != "memory":
            continue

        memory_counter += 1

        span_length = int(eeg_trial["length"])

        if span_length not in label_map:
            continue

        anchor_time = trial_anchor_time(
            eeg_trial,
            window_anchor,
        )

        window_start = (
            anchor_time
            + delay_seconds
        )

        window_stop = (
            window_start
            + duration_seconds
        )

        pupil_start = _map_eeg_time(
            window_start,
            segments,
        )

        pupil_stop = _map_eeg_time(
            window_stop,
            segments,
        )

        eeg_window = extract_padded_signal_window(
            eeg_raw,
            picks=channels,
            window_start_seconds=window_start,
            window_stop_seconds=window_stop,
            target_rate_hz=float(
                preprocessing[
                    "eeg_ecg_target_sampling_rate_hz"
                ]
            ),
            lowcut_hz=float(
                preprocessing["eeg_lowcut_hz"]
            ),
            highcut_hz=float(
                preprocessing["eeg_highcut_hz"]
            ),
            filter_order=int(
                preprocessing["filter_order"]
            ),
            notch_frequency_hz=float(
                preprocessing[
                    "power_line_frequency_hz"
                ]
            ),
            notch_quality_factor=float(
                preprocessing["notch_quality_factor"]
            ),
            padding_seconds=float(
                preprocessing["filter_padding_seconds"]
            ),
        )

        ecg_window = extract_padded_signal_window(
            ecg_raw,
            picks=["ECG"],
            window_start_seconds=window_start,
            window_stop_seconds=window_stop,
            target_rate_hz=float(
                preprocessing[
                    "eeg_ecg_target_sampling_rate_hz"
                ]
            ),
            lowcut_hz=float(
                preprocessing["ecg_lowcut_hz"]
            ),
            highcut_hz=float(
                preprocessing["ecg_highcut_hz"]
            ),
            filter_order=int(
                preprocessing["filter_order"]
            ),
            notch_frequency_hz=float(
                preprocessing[
                    "power_line_frequency_hz"
                ]
            ),
            notch_quality_factor=float(
                preprocessing["notch_quality_factor"]
            ),
            padding_seconds=float(
                preprocessing["filter_padding_seconds"]
            ),
        )

        pupil_window, pupil_quality = interpolate_pupil_window(
            pupil_streams,
            window_start_seconds=float(pupil_start),
            window_stop_seconds=float(pupil_stop),
            timestamp_field=timestamp_field,
            gaze_timestamp_field=gaze_timestamp_field,
            gaze_x_field=gaze_x_field,
            gaze_y_field=gaze_y_field,
            diameter_field=diameter_field,
            target_rate_hz=pupil_target_rate_hz,
            minimum_confidence=float(
                preprocessing[
                    "pupil_minimum_confidence"
                ]
            ),
            minimum_valid_fraction=float(
                preprocessing[
                    "pupil_minimum_valid_fraction"
                ]
            ),
            maximum_gap_seconds=float(
                preprocessing[
                    "pupil_maximum_gap_seconds"
                ]
            ),
        )

        segment_id = (
            f"{subject}_memory_"
            f"{memory_counter:03d}_"
            f"span{span_length}"
        )

        pupil_available = (
            pupil_window is not None
        )

        pupil_quality_rows.append(
            {
                "segment_id": segment_id,
                **pupil_quality,
            }
        )

        if pupil_window is None:
            pupil_window = np.full(
                (
                    expected_pupil_samples,
                    3,
                ),
                np.nan,
                dtype=float,
            )

        eeg_windows.append(eeg_window)
        ecg_windows.append(ecg_window)
        pupil_windows.append(pupil_window)

        metadata_rows.append(
            {
                "segment_id": segment_id,
                "participant": subject,
                "condition": "memory",
                "span_length": span_length,
                "label": label_map[span_length],
                "memory_trial_index": memory_counter,
                "eeg_window_start_seconds": window_start,
                "eeg_window_stop_seconds": window_stop,
                "pupil_window_start_seconds": float(
                    pupil_start
                ),
                "pupil_window_stop_seconds": float(
                    pupil_stop
                ),
                "clock_segment_count": len(segments),
                "pupil_available": bool(
                    pupil_available
                ),
                "strict_primary_cohort": bool(
                    manifest_row[
                        "strict_primary_cohort"
                    ]
                ),
            }
        )

    metadata = pd.DataFrame(metadata_rows)
    pupil_quality_frame = pd.DataFrame(
        pupil_quality_rows
    )

    if not eeg_windows:
        raise RuntimeError(
            f"{subject} produced no EEGECG windows."
        )

    arrays = {
        "eeg": np.stack(eeg_windows),
        "ecg": np.stack(ecg_windows),
        "pupil": np.stack(pupil_windows),
    }

    class_counts = Counter(
        metadata["span_length"].astype(int)
    )

    pupil_available_mask = (
        metadata["pupil_available"].astype(bool)
    )

    trimodal_class_counts = Counter(
        metadata.loc[
            pupil_available_mask,
            "span_length",
        ].astype(int)
    )

    trimodal_count = int(
        pupil_available_mask.sum()
    )

    summary = {
        "dataset": "ds003838",
        "subject": subject,
        "aligned_trial_count": int(len(alignment)),
        "memory_trial_count": int(memory_counter),
        "usable_eeg_ecg_window_count": int(
            len(metadata)
        ),
        "usable_trimodal_window_count":
            trimodal_count,
        "pupil_rejected_window_count": int(
            len(metadata) - trimodal_count
        ),
        "class_window_counts": {
            str(key): int(value)
            for key, value in sorted(class_counts.items())
        },
        "trimodal_class_window_counts": {
            str(key): int(value)
            for key, value in sorted(
                trimodal_class_counts.items()
            )
        },
        "clock_segment_count": int(len(segments)),
        "eeg_shape": list(arrays["eeg"].shape),
        "ecg_shape": list(arrays["ecg"].shape),
        "pupil_shape": list(arrays["pupil"].shape),
        "eeg_channels": channels,
        "ecg_channel": "ECG",
        "window_definition": {
            "anchor": window_anchor,
            "start_offset_seconds": delay_seconds,
            "duration_seconds": duration_seconds,
        },
        "sampling_rates_hz": {
            "eeg": float(
                preprocessing[
                    "eeg_ecg_target_sampling_rate_hz"
                ]
            ),
            "ecg": float(
                preprocessing[
                    "eeg_ecg_target_sampling_rate_hz"
                ]
            ),
            "pupil": float(
                preprocessing[
                    "pupil_target_sampling_rate_hz"
                ]
            ),
        },
        "pupil_quality_records": int(
            len(pupil_quality_frame)
        ),
    }

    return arrays, metadata, {
        "summary": summary,
        "pupil_quality": pupil_quality_frame,
    }


def write_subject_output(
    arrays: dict[str, np.ndarray],
    metadata: pd.DataFrame,
    audit: dict[str, Any],
    output_directory: str | Path,
) -> Path:
    """Write one subject atomically without overwriting outputs."""

    destination = Path(output_directory)

    if destination.exists():
        raise FileExistsError(
            f"Refusing to overwrite subject output: {destination}"
        )

    temporary = destination.with_name(
        destination.name + ".building"
    )

    if temporary.exists():
        raise FileExistsError(
            f"Temporary output already exists: {temporary}"
        )

    temporary.mkdir(
        parents=True,
        exist_ok=False,
    )

    np.savez_compressed(
        temporary / "windows.npz",
        eeg=arrays["eeg"],
        ecg=arrays["ecg"],
        pupil=arrays["pupil"],
    )

    metadata.to_csv(
        temporary / "metadata.csv",
        index=False,
    )

    audit["pupil_quality"].to_csv(
        temporary / "pupil_quality.csv",
        index=False,
    )

    (
        temporary / "summary.json"
    ).write_text(
        json.dumps(
            audit["summary"],
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )

    temporary.replace(destination)

    return destination


def run_smoke_subject(
    config: dict[str, Any],
    *,
    subject: str,
    write_outputs: bool,
) -> dict[str, Any]:
    """Run one validated strict-primary subject."""

    manifest_path = resolve_project_path(
        "outputs/revision/manifests/"
        "ds003838/ds003838_subjects.csv"
    )

    manifest = pd.read_csv(
        manifest_path,
        low_memory=False,
    )

    matches = manifest.loc[
        manifest["subject"].astype(str) == str(subject)
    ]

    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one manifest row for {subject}."
        )

    row = matches.iloc[0]

    if str(row["strict_primary_cohort"]).lower() not in {
        "true",
        "1",
    }:
        raise RuntimeError(
            f"{subject} is not a strict-primary subject."
        )

    arrays, metadata, audit = preprocess_subject(
        row,
        config,
    )

    if write_outputs:
        destination = resolve_project_path(
            "outputs/revision/preprocessed/"
            f"ds003838_smoke/{subject}"
        )

        write_subject_output(
            arrays,
            metadata,
            audit,
            destination,
        )

        audit["summary"]["output_directory"] = str(
            destination
        )

    print(
        json.dumps(
            audit["summary"],
            indent=2,
            sort_keys=True,
        )
    )

    return audit["summary"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Preprocess one strict-primary ds003838 "
            "participant as a smoke test."
        )
    )

    parser.add_argument(
        "--subject",
        default="sub-033",
    )

    parser.add_argument(
        "--write",
        action="store_true",
    )

    arguments = parser.parse_args()

    config = load_revision_config(
        check_input_paths=True
    )

    run_smoke_subject(
        config,
        subject=arguments.subject,
        write_outputs=arguments.write,
    )


if __name__ == "__main__":
    main()