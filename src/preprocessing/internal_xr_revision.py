"""Corrected preprocessing for the internal XR attention dataset.

This revision-only module:

1. Uses all 13 participants.
2. Retains exactly eight manuscript EEG channels.
3. Applies zero-phase 0.5--40 Hz filtering to EEG and ECG.
4. Resamples EEG and ECG from 512 Hz to 128 Hz.
5. Preserves pupil data at its native 30 Hz sampling rate.
6. Converts pupil radius values <= 0 to tracking loss.
7. Interpolates only tracking-loss gaps no longer than 0.5 seconds.
8. Creates aligned, non-overlapping four-second windows.
9. Preserves all EEG/ECG windows and records pupil availability separately.
10. Performs no global normalization or feature selection.

Standardization and feature selection must be fitted using training data only.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import butter, resample_poly, sosfiltfilt

from src.core.config import load_revision_config


PROJECT_ROOT = Path(__file__).resolve().parents[2]

FILENAME_PATTERN = re.compile(
    r"^(?P<phase>[123])\."
    r"(?P<participant>P\d{2})_"
    r"(?P<modality>EEG|ECG|Pupil)\.csv$",
    re.IGNORECASE,
)

ORIGINAL_EEG_ECG_RATE_HZ = 512.0
EXPECTED_PUPIL_RATE_HZ = 30.0


def parse_recording_filename(
    filename: str,
) -> tuple[int, str, str]:
    """Parse one internal XR raw-data filename."""

    match = FILENAME_PATTERN.fullmatch(filename)

    if match is None:
        raise ValueError(
            f"Unexpected internal XR filename: {filename}"
        )

    phase = int(match.group("phase"))
    participant = match.group("participant").upper()
    raw_modality = match.group("modality")

    modality = (
        "Pupil"
        if raw_modality.casefold() == "pupil"
        else raw_modality.upper()
    )

    return phase, participant, modality


def contiguous_true_runs(
    mask: np.ndarray,
) -> list[tuple[int, int]]:
    """Return half-open intervals for contiguous True runs."""

    boolean_mask = np.asarray(mask, dtype=bool)

    if boolean_mask.ndim != 1:
        raise ValueError("mask must be one-dimensional")

    padded = np.concatenate(
        ([False], boolean_mask, [False])
    )

    transitions = np.flatnonzero(
        padded[1:] != padded[:-1]
    )

    return list(
        zip(
            transitions[::2].tolist(),
            transitions[1::2].tolist(),
        )
    )


def run_length_by_sample(
    mask: np.ndarray,
) -> np.ndarray:
    """Assign each invalid sample the full length of its run."""

    result = np.zeros(
        len(mask),
        dtype=np.int32,
    )

    for start, stop in contiguous_true_runs(mask):
        result[start:stop] = stop - start

    return result


def interpolate_short_tracking_gaps(
    values: np.ndarray,
    invalid_mask: np.ndarray,
    maximum_gap_samples: int,
) -> np.ndarray:
    """Interpolate only complete tracking-loss runs within the threshold.

    Internal gaps are linearly interpolated. Short gaps at the beginning
    or end of a recording use the nearest observed value. Long gaps remain
    missing and therefore make affected pupil windows unavailable.
    """

    array = np.asarray(values, dtype=float)

    if array.ndim == 1:
        array = array[:, np.newaxis]

    invalid = np.asarray(
        invalid_mask,
        dtype=bool,
    )

    if len(array) != len(invalid):
        raise ValueError(
            "values and invalid_mask must have equal length"
        )

    cleaned = array.copy()
    cleaned[invalid, :] = np.nan

    row_count = len(cleaned)

    for start, stop in contiguous_true_runs(invalid):
        gap_length = stop - start

        if gap_length > maximum_gap_samples:
            continue

        left_index = start - 1
        right_index = stop

        has_left = (
            left_index >= 0
            and np.isfinite(cleaned[left_index]).all()
        )

        has_right = (
            right_index < row_count
            and np.isfinite(cleaned[right_index]).all()
        )

        if has_left and has_right:
            fractions = (
                np.arange(
                    1,
                    gap_length + 1,
                    dtype=float,
                )
                / (gap_length + 1)
            )[:, np.newaxis]

            left = cleaned[left_index]
            right = cleaned[right_index]

            cleaned[start:stop] = (
                left
                + fractions * (right - left)
            )

        elif has_right:
            cleaned[start:stop] = cleaned[right_index]

        elif has_left:
            cleaned[start:stop] = cleaned[left_index]

    return cleaned


def pupil_window_available(
    raw_invalid_mask: np.ndarray,
    full_run_lengths: np.ndarray,
    minimum_valid_fraction: float,
    maximum_gap_samples: int,
) -> bool:
    """Apply the locked pupil-window quality rule."""

    invalid = np.asarray(
        raw_invalid_mask,
        dtype=bool,
    )

    runs = np.asarray(
        full_run_lengths,
        dtype=np.int32,
    )

    if len(invalid) != len(runs):
        raise ValueError(
            "raw_invalid_mask and full_run_lengths "
            "must have equal length"
        )

    valid_fraction = float(
        1.0 - invalid.mean()
    )

    longest_gap = int(
        runs.max()
        if len(runs)
        else 0
    )

    return bool(
        valid_fraction >= minimum_valid_fraction
        and longest_gap <= maximum_gap_samples
    )


def filter_and_resample(
    signal: np.ndarray,
    original_rate_hz: float,
    target_rate_hz: float,
    lowcut_hz: float,
    highcut_hz: float,
    filter_order: int,
) -> np.ndarray:
    """Bandpass-filter and polyphase-resample a signal array."""

    array = np.asarray(
        signal,
        dtype=float,
    )

    if array.ndim == 1:
        array = array[:, np.newaxis]

    if not np.isfinite(array).all():
        raise ValueError(
            "EEG/ECG input contains non-finite values"
        )

    sos = butter(
        filter_order,
        [lowcut_hz, highcut_hz],
        btype="bandpass",
        fs=original_rate_hz,
        output="sos",
    )

    filtered = sosfiltfilt(
        sos,
        array,
        axis=0,
    )

    ratio = (
        target_rate_hz
        / original_rate_hz
    )

    if np.isclose(ratio, 0.25):
        up, down = 1, 4
    else:
        raise ValueError(
            "This validated internal pipeline expects "
            "512 Hz to 128 Hz resampling."
        )

    resampled = resample_poly(
        filtered,
        up=up,
        down=down,
        axis=0,
    )

    if not np.isfinite(resampled).all():
        raise RuntimeError(
            "Filtering/resampling produced non-finite values"
        )

    return resampled


def window_array(
    signal: np.ndarray,
    samples_per_window: int,
) -> np.ndarray:
    """Create non-overlapping windows without dropping samples silently."""

    array = np.asarray(signal)

    if len(array) % samples_per_window != 0:
        raise ValueError(
            f"Signal length {len(array)} is not divisible by "
            f"window length {samples_per_window}."
        )

    window_count = (
        len(array)
        // samples_per_window
    )

    return array.reshape(
        window_count,
        samples_per_window,
        *array.shape[1:],
    )


def infer_sampling_rate(
    time_values: np.ndarray,
) -> float:
    """Infer the median sampling rate from timestamps."""

    time = np.asarray(
        time_values,
        dtype=float,
    )

    differences = np.diff(time)

    if (
        len(differences) == 0
        or not np.isfinite(differences).all()
        or np.any(differences <= 0)
    ):
        raise ValueError(
            "Timestamps must be finite and strictly increasing"
        )

    return float(
        1.0 / np.median(differences)
    )


def resolve_output_directory(
    config: dict[str, Any],
    explicit_output: str | None,
) -> Path:
    """Resolve the revision preprocessing output directory."""

    if explicit_output is not None:
        path = Path(explicit_output)
    else:
        configured = config[
            "datasets"
        ][
            "internal_xr"
        ].get(
            "output_dir",
            "outputs/revision/preprocessed/internal_xr",
        )

        path = Path(configured)

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path.resolve()


def build_internal_xr_preprocessing(
    config: dict[str, Any],
    *,
    write_outputs: bool,
    explicit_output: str | None = None,
) -> dict[str, Any]:
    """Run corrected preprocessing and optionally save its artifacts."""

    raw_root = Path(
        config["paths"]["internal_xr_raw"]
    ).resolve()

    if not raw_root.is_dir():
        raise FileNotFoundError(
            f"Internal XR raw-data root not found: {raw_root}"
        )

    dataset_config = config[
        "datasets"
    ][
        "internal_xr"
    ]

    preprocessing = config["preprocessing"]

    participants = [
        str(participant).upper()
        for participant in dataset_config["participants"]
    ]

    eeg_channels = list(
        config["eeg"]["internal_channels"]
    )

    phase_labels = {
        phase: int(
            dataset_config[
                "phase_labels"
            ][
                f"phase_{phase}"
            ]
        )
        for phase in (1, 2, 3)
    }

    target_rate_hz = float(
        preprocessing.get(
            "internal_target_sampling_rate_hz",
            128.0,
        )
    )

    window_seconds = float(
        preprocessing.get(
            "internal_window_seconds",
            4.0,
        )
    )

    filter_order = int(
        preprocessing.get(
            "filter_order",
            5,
        )
    )

    minimum_pupil_valid_fraction = float(
        preprocessing.get(
            "pupil_minimum_valid_fraction",
            0.80,
        )
    )

    maximum_pupil_gap_seconds = float(
        preprocessing.get(
            "pupil_maximum_gap_seconds",
            0.50,
        )
    )

    eeg_lowcut, eeg_highcut = map(
        float,
        preprocessing["eeg_bandpass_hz"],
    )

    ecg_lowcut, ecg_highcut = map(
        float,
        preprocessing["ecg_bandpass_hz"],
    )

    output_directory = resolve_output_directory(
        config,
        explicit_output,
    )

    temporary_directory = output_directory.with_name(
        output_directory.name + ".building"
    )

    if write_outputs:
        if output_directory.exists():
            raise FileExistsError(
                "Refusing to overwrite existing corrected output: "
                f"{output_directory}"
            )

        if temporary_directory.exists():
            raise FileExistsError(
                "A previous temporary preprocessing directory exists: "
                f"{temporary_directory}"
            )

        recordings_directory = (
            temporary_directory
            / "recordings"
        )

        recordings_directory.mkdir(
            parents=True,
            exist_ok=False,
        )
    else:
        recordings_directory = None

    manifest_rows: list[dict[str, Any]] = []
    recording_summaries: list[dict[str, Any]] = []

    eeg_samples_per_window = int(
        round(
            target_rate_hz
            * window_seconds
        )
    )

    for participant in participants:
        for phase in (1, 2, 3):
            eeg_filename = (
                f"{phase}.{participant}_EEG.csv"
            )

            ecg_filename = (
                f"{phase}.{participant}_ECG.csv"
            )

            pupil_filename = (
                f"{phase}.{participant}_Pupil.csv"
            )

            eeg_path = (
                raw_root
                / "EEG"
                / eeg_filename
            )

            ecg_path = (
                raw_root
                / "ECG"
                / ecg_filename
            )

            pupil_path = (
                raw_root
                / "Pupil"
                / pupil_filename
            )

            for expected_path, expected_modality in (
                (eeg_path, "EEG"),
                (ecg_path, "ECG"),
                (pupil_path, "Pupil"),
            ):
                if not expected_path.is_file():
                    raise FileNotFoundError(
                        f"Missing {expected_modality} file: "
                        f"{expected_path}"
                    )

                parsed_phase, parsed_participant, parsed_modality = (
                    parse_recording_filename(
                        expected_path.name
                    )
                )

                if (
                    parsed_phase != phase
                    or parsed_participant != participant
                    or parsed_modality != expected_modality
                ):
                    raise RuntimeError(
                        f"Filename metadata mismatch: {expected_path}"
                    )

            eeg_frame = pd.read_csv(
                eeg_path,
                usecols=["Time", *eeg_channels],
                encoding="utf-8-sig",
                low_memory=False,
            )

            ecg_frame = pd.read_csv(
                ecg_path,
                usecols=["Time", "ECG_Signal"],
                encoding="utf-8-sig",
                low_memory=False,
            )

            pupil_frame = pd.read_csv(
                pupil_path,
                usecols=["Time", "x", "y", "r"],
                encoding="utf-8-sig",
                low_memory=False,
            )

            eeg_rate = infer_sampling_rate(
                eeg_frame["Time"].to_numpy(
                    dtype=float
                )
            )

            ecg_rate = infer_sampling_rate(
                ecg_frame["Time"].to_numpy(
                    dtype=float
                )
            )

            pupil_rate = infer_sampling_rate(
                pupil_frame["Time"].to_numpy(
                    dtype=float
                )
            )

            if abs(eeg_rate - ORIGINAL_EEG_ECG_RATE_HZ) > 0.1:
                raise RuntimeError(
                    f"Unexpected EEG rate for {participant}, "
                    f"phase {phase}: {eeg_rate}"
                )

            # ECG timestamps are rounded to six decimals in phases 1 and 3,
            # so the nominal acquisition rate is used after validating that
            # the inferred rate remains within 0.1 Hz.
            if abs(ecg_rate - ORIGINAL_EEG_ECG_RATE_HZ) > 0.1:
                raise RuntimeError(
                    f"Unexpected ECG rate for {participant}, "
                    f"phase {phase}: {ecg_rate}"
                )

            if abs(pupil_rate - EXPECTED_PUPIL_RATE_HZ) > 0.1:
                raise RuntimeError(
                    f"Unexpected pupil rate for {participant}, "
                    f"phase {phase}: {pupil_rate}"
                )

            eeg_signal = eeg_frame[
                eeg_channels
            ].to_numpy(dtype=float)

            ecg_signal = ecg_frame[
                ["ECG_Signal"]
            ].to_numpy(dtype=float)

            eeg_resampled = filter_and_resample(
                eeg_signal,
                original_rate_hz=ORIGINAL_EEG_ECG_RATE_HZ,
                target_rate_hz=target_rate_hz,
                lowcut_hz=eeg_lowcut,
                highcut_hz=eeg_highcut,
                filter_order=filter_order,
            )

            ecg_resampled = filter_and_resample(
                ecg_signal,
                original_rate_hz=ORIGINAL_EEG_ECG_RATE_HZ,
                target_rate_hz=target_rate_hz,
                lowcut_hz=ecg_lowcut,
                highcut_hz=ecg_highcut,
                filter_order=filter_order,
            )

            eeg_windows = window_array(
                eeg_resampled,
                eeg_samples_per_window,
            )

            ecg_windows = window_array(
                ecg_resampled,
                eeg_samples_per_window,
            )

            pupil_samples_per_window = int(
                round(
                    EXPECTED_PUPIL_RATE_HZ
                    * window_seconds
                )
            )

            pupil_values = pupil_frame[
                ["x", "y", "r"]
            ].to_numpy(dtype=float)

            raw_pupil_invalid = (
                ~np.isfinite(
                    pupil_values[:, 2]
                )
                | (
                    pupil_values[:, 2]
                    <= 0
                )
            )

            maximum_pupil_gap_samples = int(
                round(
                    maximum_pupil_gap_seconds
                    * EXPECTED_PUPIL_RATE_HZ
                )
            )

            pupil_cleaned = (
                interpolate_short_tracking_gaps(
                    pupil_values,
                    raw_pupil_invalid,
                    maximum_pupil_gap_samples,
                )
            )

            pupil_windows = window_array(
                pupil_cleaned,
                pupil_samples_per_window,
            )

            pupil_invalid_windows = window_array(
                raw_pupil_invalid,
                pupil_samples_per_window,
            )

            full_run_lengths = run_length_by_sample(
                raw_pupil_invalid
            )

            pupil_run_windows = window_array(
                full_run_lengths,
                pupil_samples_per_window,
            )

            window_count = len(eeg_windows)

            if not (
                len(ecg_windows)
                == len(pupil_windows)
                == window_count
            ):
                raise RuntimeError(
                    f"Cross-modal window mismatch for "
                    f"{participant}, phase {phase}: "
                    f"EEG={len(eeg_windows)}, "
                    f"ECG={len(ecg_windows)}, "
                    f"Pupil={len(pupil_windows)}"
                )

            pupil_available = np.zeros(
                window_count,
                dtype=bool,
            )

            pupil_valid_fraction = np.zeros(
                window_count,
                dtype=np.float32,
            )

            pupil_longest_gap_seconds = np.zeros(
                window_count,
                dtype=np.float32,
            )

            label = phase_labels[phase]

            for window_index in range(window_count):
                invalid_window = (
                    pupil_invalid_windows[
                        window_index
                    ]
                )

                run_window = (
                    pupil_run_windows[
                        window_index
                    ]
                )

                valid_fraction = float(
                    1.0
                    - invalid_window.mean()
                )

                longest_gap_seconds = float(
                    (
                        int(run_window.max())
                        if len(run_window)
                        else 0
                    )
                    / EXPECTED_PUPIL_RATE_HZ
                )

                available = (
                    pupil_window_available(
                        invalid_window,
                        run_window,
                        minimum_pupil_valid_fraction,
                        maximum_pupil_gap_samples,
                    )
                    and np.isfinite(
                        pupil_windows[
                            window_index
                        ]
                    ).all()
                )

                pupil_available[
                    window_index
                ] = available

                pupil_valid_fraction[
                    window_index
                ] = valid_fraction

                pupil_longest_gap_seconds[
                    window_index
                ] = longest_gap_seconds

                segment_id = (
                    f"{participant}_"
                    f"phase{phase}_"
                    f"window{window_index:03d}"
                )

                manifest_rows.append(
                    {
                        "segment_id": segment_id,
                        "participant": participant,
                        "phase": phase,
                        "label": label,
                        "window_index": window_index,
                        "window_start_seconds":
                            window_index
                            * window_seconds,
                        "window_end_seconds":
                            (
                                window_index + 1
                            )
                            * window_seconds,
                        "eeg_available": True,
                        "ecg_available": True,
                        "pupil_available":
                            bool(available),
                        "pupil_raw_valid_fraction":
                            valid_fraction,
                        "pupil_longest_gap_seconds":
                            longest_gap_seconds,
                        "recording_archive":
                            str(
                                Path("recordings")
                                / (
                                    f"{participant}_"
                                    f"phase{phase}.npz"
                                )
                            ),
                        "archive_window_index":
                            window_index,
                        "eeg_source":
                            str(
                                eeg_path.relative_to(
                                    raw_root
                                )
                            ),
                        "ecg_source":
                            str(
                                ecg_path.relative_to(
                                    raw_root
                                )
                            ),
                        "pupil_source":
                            str(
                                pupil_path.relative_to(
                                    raw_root
                                )
                            ),
                    }
                )

            if write_outputs:
                archive_path = (
                    recordings_directory
                    / (
                        f"{participant}_"
                        f"phase{phase}.npz"
                    )
                )

                np.savez_compressed(
                    archive_path,
                    eeg=eeg_windows.astype(
                        np.float32,
                        copy=False,
                    ),
                    ecg=ecg_windows[
                        :,
                        :,
                        0,
                    ].astype(
                        np.float32,
                        copy=False,
                    ),
                    pupil=pupil_windows.astype(
                        np.float32,
                        copy=False,
                    ),
                    pupil_available=pupil_available,
                    pupil_raw_valid_fraction=(
                        pupil_valid_fraction
                    ),
                    pupil_longest_gap_seconds=(
                        pupil_longest_gap_seconds
                    ),
                    eeg_channels=np.asarray(
                        eeg_channels,
                        dtype="U16",
                    ),
                    pupil_columns=np.asarray(
                        ["x", "y", "r"],
                        dtype="U8",
                    ),
                    label=np.asarray(
                        label,
                        dtype=np.int16,
                    ),
                    phase=np.asarray(
                        phase,
                        dtype=np.int16,
                    ),
                    window_seconds=np.asarray(
                        window_seconds,
                        dtype=np.float32,
                    ),
                    eeg_ecg_sampling_rate_hz=(
                        np.asarray(
                            target_rate_hz,
                            dtype=np.float32,
                        )
                    ),
                    pupil_sampling_rate_hz=(
                        np.asarray(
                            EXPECTED_PUPIL_RATE_HZ,
                            dtype=np.float32,
                        )
                    ),
                )

            recording_summaries.append(
                {
                    "participant": participant,
                    "phase": phase,
                    "label": label,
                    "windows": window_count,
                    "pupil_available_windows":
                        int(
                            pupil_available.sum()
                        ),
                    "pupil_unavailable_windows":
                        int(
                            (
                                ~pupil_available
                            ).sum()
                        ),
                }
            )

            print(
                f"Processed {participant} phase {phase}: "
                f"{window_count} windows, "
                f"{int(pupil_available.sum())} "
                f"pupil-valid",
                flush=True,
            )

    manifest = pd.DataFrame(
        manifest_rows
    )

    expected_total_windows = (
        len(participants)
        * (
            75
            + 105
            + 75
        )
    )

    if len(manifest) != expected_total_windows:
        raise RuntimeError(
            f"Expected {expected_total_windows} windows, "
            f"obtained {len(manifest)}."
        )

    if manifest["segment_id"].duplicated().any():
        raise RuntimeError(
            "Duplicate segment identifiers detected"
        )

    label_counts = {
        str(int(label)): int(count)
        for label, count in (
            manifest["label"]
            .value_counts()
            .sort_index()
            .items()
        )
    }

    pupil_label_counts = {
        str(int(label)): int(count)
        for label, count in (
            manifest.loc[
                manifest["pupil_available"],
                "label",
            ]
            .value_counts()
            .reindex(
                sorted(
                    manifest[
                        "label"
                    ].unique()
                ),
                fill_value=0,
            )
            .sort_index()
            .items()
        )
    }

    summary = {
        "dataset": "internal_xr_attention",
        "participants": participants,
        "participant_count": len(participants),
        "phases": [1, 2, 3],
        "phase_label_map": {
            str(phase): label
            for phase, label in phase_labels.items()
        },
        "eeg_channels": eeg_channels,
        "eeg_channel_count": len(eeg_channels),
        "window_seconds": window_seconds,
        "eeg_ecg_original_rate_hz":
            ORIGINAL_EEG_ECG_RATE_HZ,
        "eeg_ecg_processed_rate_hz":
            target_rate_hz,
        "pupil_native_rate_hz":
            EXPECTED_PUPIL_RATE_HZ,
        "eeg_bandpass_hz": [
            eeg_lowcut,
            eeg_highcut,
        ],
        "ecg_bandpass_hz": [
            ecg_lowcut,
            ecg_highcut,
        ],
        "filter_order": filter_order,
        "notch_applied": False,
        "notch_rationale":
            "The 40 Hz low-pass cutoff is below "
            "the 60 Hz line frequency.",
        "standardization_applied": False,
        "standardization_policy":
            "Fit using training data only.",
        "total_windows": int(
            len(manifest)
        ),
        "label_counts": label_counts,
        "pupil_quality_rule": {
            "minimum_raw_valid_fraction":
                minimum_pupil_valid_fraction,
            "maximum_tracking_loss_gap_seconds":
                maximum_pupil_gap_seconds,
            "invalid_definition":
                "non-finite radius or radius <= 0",
        },
        "pupil_available_windows": int(
            manifest[
                "pupil_available"
            ].sum()
        ),
        "pupil_unavailable_windows": int(
            (
                ~manifest[
                    "pupil_available"
                ]
            ).sum()
        ),
        "pupil_available_label_counts":
            pupil_label_counts,
        "recording_count": len(
            recording_summaries
        ),
        "recordings": recording_summaries,
    }

    if write_outputs:
        manifest.to_csv(
            temporary_directory
            / "window_manifest.csv",
            index=False,
        )

        (
            temporary_directory
            / "preprocessing_summary.json"
        ).write_text(
            json.dumps(
                summary,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        resolved_settings = {
            "raw_root": str(raw_root),
            "output_directory":
                str(output_directory),
            "parameters": {
                "target_rate_hz":
                    target_rate_hz,
                "window_seconds":
                    window_seconds,
                "filter_order":
                    filter_order,
                "minimum_pupil_valid_fraction":
                    minimum_pupil_valid_fraction,
                "maximum_pupil_gap_seconds":
                    maximum_pupil_gap_seconds,
                "eeg_bandpass_hz": [
                    eeg_lowcut,
                    eeg_highcut,
                ],
                "ecg_bandpass_hz": [
                    ecg_lowcut,
                    ecg_highcut,
                ],
            },
        }

        (
            temporary_directory
            / "resolved_preprocessing_config.json"
        ).write_text(
            json.dumps(
                resolved_settings,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        temporary_directory.replace(
            output_directory
        )

    print("\n===== INTERNAL XR PREPROCESSING SUMMARY =====")
    print("Participants:", summary["participant_count"])
    print("Recordings:", summary["recording_count"])
    print("Total windows:", summary["total_windows"])
    print("Label counts:", summary["label_counts"])
    print(
        "Pupil-available windows:",
        summary["pupil_available_windows"],
    )
    print(
        "Pupil-unavailable windows:",
        summary["pupil_unavailable_windows"],
    )
    print(
        "Pupil-available label counts:",
        summary[
            "pupil_available_label_counts"
        ],
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
            "Build corrected internal XR "
            "preprocessing artifacts."
        )
    )

    parser.add_argument(
        "--write",
        action="store_true",
        help="Write corrected arrays and manifests.",
    )

    parser.add_argument(
        "--output-root",
        default=None,
        help=(
            "Optional output-directory override. "
            "Existing output is never overwritten."
        ),
    )

    arguments = parser.parse_args()

    config = load_revision_config(
        check_input_paths=True
    )

    build_internal_xr_preprocessing(
        config,
        write_outputs=arguments.write,
        explicit_output=arguments.output_root,
    )


if __name__ == "__main__":
    main()