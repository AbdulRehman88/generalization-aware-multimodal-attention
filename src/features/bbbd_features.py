"""Leakage-safe BBBD multimodal feature construction."""

from __future__ import annotations

import argparse
import gzip
import io
import json
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import mne
import numpy as np
import pandas as pd
import yaml
from scipy.signal import butter, sosfiltfilt, welch

from src.core.config import load_revision_config
from src.features.internal_xr_revision_features import (
    extract_ecg_features,
    extract_eeg_features,
    signal_shape_features,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

METADATA_COLUMNS = [
    "segment_id",
    "recording_id",
    "dataset",
    "participant",
    "subject",
    "session",
    "task",
    "label",
    "condition",
    "window_index",
    "start_seconds",
    "end_seconds",
    "pupil_valid_fraction",
]

MODALITY_TABLES = [
    "EEG",
    "ECG",
    "Pupil",
    "ECG_EEG",
    "ECG_Pupil",
    "EEG_Pupil",
    "ECG_EEG_Pupil",
]


@dataclass(frozen=True)
class RecordingRequest:
    """Identify one BBBD recording."""

    dataset: str
    subject: str
    session: str
    task: str


def resolve_project_path(
    value: str | Path,
) -> Path:
    """Resolve a project-relative path."""

    path = Path(value)

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path.resolve()


def load_bbbd_config(
    path: str | Path = "configs/bbbd.yaml",
) -> dict[str, Any]:
    """Load and minimally validate the BBBD policy configuration."""

    resolved = resolve_project_path(path)

    if not resolved.is_file():
        raise FileNotFoundError(
            f"BBBD configuration missing: {resolved}"
        )

    payload = yaml.safe_load(
        resolved.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(payload, dict):
        raise TypeError(
            "BBBD configuration must be a mapping."
        )

    required = {
        "study",
        "datasets",
        "participant_identity",
        "signals",
        "windowing",
        "features",
        "outputs",
    }

    missing = required - set(payload)

    if missing:
        raise ValueError(
            f"BBBD configuration lacks sections: {sorted(missing)}"
        )

    return payload


def namespaced_participant(
    dataset: str,
    subject: str,
) -> str:
    """Create the experiment-qualified participant identity."""

    return f"{str(dataset)}::{str(subject)}"


def robust_boolean_mask(
    values: pd.Series,
) -> pd.Series:
    """Interpret common boolean representations safely."""

    return (
        values.astype(str)
        .str.strip()
        .str.lower()
        .isin(
            {
                "true",
                "1",
                "yes",
                "y",
            }
        )
    )



def load_direct_cohort_participants(
    config: dict[str, Any],
) -> set[str]:
    """Load and validate the locked direct-comparison cohort."""

    cohort_config = config[
        "cohorts"
    ][
        "direct_seven_path"
    ]

    registry_path = resolve_project_path(
        cohort_config[
            "registry"
        ]
    )

    if not registry_path.is_file():
        raise FileNotFoundError(
            f"BBBD cohort registry missing: {registry_path}"
        )

    registry = pd.read_csv(
        registry_path,
        low_memory=False,
    )

    selection_column = str(
        cohort_config[
            "selection_column"
        ]
    )

    required_columns = {
        "dataset",
        "participant",
        "recordings",
        selection_column,
    }

    missing = (
        required_columns
        - set(
            registry.columns
        )
    )

    if missing:
        raise RuntimeError(
            "BBBD cohort registry lacks columns: "
            f"{sorted(missing)}"
        )

    selected = registry.loc[
        robust_boolean_mask(
            registry[
                selection_column
            ]
        )
    ].copy()

    expected_participants = int(
        cohort_config[
            "expected_participants"
        ]
    )

    expected_recordings = int(
        cohort_config[
            "expected_recordings"
        ]
    )

    if len(
        selected
    ) != expected_participants:
        raise RuntimeError(
            "BBBD selected participant count differs from "
            f"the locked value: {len(selected)} versus "
            f"{expected_participants}."
        )

    observed_recordings = int(
        pd.to_numeric(
            selected[
                "recordings"
            ],
            errors="raise",
        ).sum()
    )

    if observed_recordings != expected_recordings:
        raise RuntimeError(
            "BBBD selected recording count differs from "
            f"the locked value: {observed_recordings} versus "
            f"{expected_recordings}."
        )

    participants = set(
        selected[
            "participant"
        ].astype(str)
    )

    if len(
        participants
    ) != expected_participants:
        raise RuntimeError(
            "Duplicate selected participant identities detected."
        )

    return participants


def window_start_indices(
    start_sample: int,
    stop_sample: int,
    window_samples: int,
    step_samples: int,
) -> list[int]:
    """Return complete deterministic window starts."""

    start_sample = int(start_sample)
    stop_sample = int(stop_sample)
    window_samples = int(window_samples)
    step_samples = int(step_samples)

    if start_sample < 0:
        raise ValueError(
            "start_sample cannot be negative."
        )

    if stop_sample < start_sample:
        raise ValueError(
            "stop_sample cannot precede start_sample."
        )

    if window_samples <= 0:
        raise ValueError(
            "window_samples must be positive."
        )

    if step_samples <= 0:
        raise ValueError(
            "step_samples must be positive."
        )

    last_start = (
        stop_sample
        - window_samples
    )

    if last_start < start_sample:
        return []

    return list(
        range(
            start_sample,
            last_start + 1,
            step_samples,
        )
    )


def contiguous_false_runs(
    mask: np.ndarray,
) -> list[tuple[int, int]]:
    """Return half-open runs where a boolean mask is false."""

    values = np.asarray(
        mask,
        dtype=bool,
    )

    if values.ndim != 1:
        raise ValueError(
            "Mask must be one-dimensional."
        )

    false_values = ~values

    if not false_values.any():
        return []

    changes = np.diff(
        np.concatenate(
            (
                np.array([False]),
                false_values,
                np.array([False]),
            )
        ).astype(int)
    )

    starts = np.flatnonzero(
        changes == 1
    )

    stops = np.flatnonzero(
        changes == -1
    )

    return [
        (
            int(start),
            int(stop),
        )
        for start, stop
        in zip(
            starts,
            stops,
        )
    ]


def interpolate_short_invalid_runs(
    values: np.ndarray,
    valid_mask: np.ndarray,
    *,
    maximum_gap_samples: int,
) -> np.ndarray:
    """Interpolate only bounded invalid runs no longer than the limit."""

    array = np.asarray(
        values,
        dtype=float,
    ).copy()

    valid = np.asarray(
        valid_mask,
        dtype=bool,
    )

    if array.ndim != 1:
        raise ValueError(
            "Signal must be one-dimensional."
        )

    if valid.shape != array.shape:
        raise ValueError(
            "Signal and validity mask shapes differ."
        )

    maximum_gap_samples = int(
        maximum_gap_samples
    )

    if maximum_gap_samples < 0:
        raise ValueError(
            "maximum_gap_samples cannot be negative."
        )

    array[
        ~valid
    ] = np.nan

    interpolated = (
        pd.Series(array)
        .interpolate(
            method="linear",
            limit_area="inside",
        )
        .to_numpy(dtype=float)
    )

    for start, stop in contiguous_false_runs(
        valid
    ):
        gap_length = (
            stop
            - start
        )

        bounded = (
            start > 0
            and stop < len(array)
        )

        if (
            not bounded
            or gap_length > maximum_gap_samples
        ):
            interpolated[
                start:stop
            ] = np.nan

    return interpolated


def preprocess_pupil(
    values: np.ndarray,
    *,
    minimum_valid_value_exclusive: float,
    maximum_gap_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate and interpolate pupil size without filling long gaps."""

    array = np.asarray(
        values,
        dtype=float,
    ).reshape(-1)

    valid = (
        np.isfinite(array)
        & (
            array
            > float(
                minimum_valid_value_exclusive
            )
        )
    )

    cleaned = interpolate_short_invalid_runs(
        array,
        valid,
        maximum_gap_samples=
            maximum_gap_samples,
    )

    return cleaned, valid


def interpolate_all_internal_nonfinite(
    values: np.ndarray,
) -> np.ndarray:
    """Interpolate internal nonfinite ECG samples."""

    array = np.asarray(
        values,
        dtype=float,
    ).reshape(-1)

    finite = np.isfinite(
        array
    )

    if finite.all():
        return array.copy()

    if finite.sum() < 2:
        raise ValueError(
            "Signal has fewer than two finite samples."
        )

    indices = np.arange(
        len(array),
        dtype=float,
    )

    cleaned = np.interp(
        indices,
        indices[
            finite
        ],
        array[
            finite
        ],
    )

    return cleaned


def preprocess_ecg(
    values: np.ndarray,
    *,
    sampling_frequency_hz: float,
    bandpass_hz: Iterable[float],
    filter_order: int,
    minimum_finite_fraction: float,
) -> np.ndarray:
    """Apply one uniform deterministic raw-ECG preprocessing policy."""

    array = np.asarray(
        values,
        dtype=float,
    ).reshape(-1)

    finite_fraction = float(
        np.mean(
            np.isfinite(
                array
            )
        )
    )

    if finite_fraction < float(
        minimum_finite_fraction
    ):
        raise ValueError(
            "ECG finite fraction is below the configured minimum: "
            f"{finite_fraction:.6f}"
        )

    array = interpolate_all_internal_nonfinite(
        array
    )

    array = (
        array
        - np.median(
            array
        )
    )

    lowcut, highcut = [
        float(value)
        for value in bandpass_hz
    ]

    nyquist = (
        float(
            sampling_frequency_hz
        )
        / 2.0
    )

    if not (
        0.0
        < lowcut
        < highcut
        < nyquist
    ):
        raise ValueError(
            "Invalid ECG bandpass frequencies."
        )

    sos = butter(
        int(filter_order),
        [
            lowcut / nyquist,
            highcut / nyquist,
        ],
        btype="bandpass",
        output="sos",
    )

    filtered = sosfiltfilt(
        sos,
        array,
    )

    if not np.isfinite(
        filtered
    ).all():
        raise RuntimeError(
            "Filtered ECG contains nonfinite values."
        )

    return filtered


def pupil_features(
    window: np.ndarray,
    *,
    sampling_frequency_hz: float,
    bands_hz: dict[str, list[float]],
    nperseg: int,
) -> dict[str, float]:
    """Extract pupil-size features without implying gaze availability."""

    values = np.asarray(
        window,
        dtype=float,
    ).reshape(-1)

    if not np.isfinite(
        values
    ).all():
        raise ValueError(
            "Pupil window contains nonfinite values."
        )

    features = {
        f"shape_{key}":
            float(value)
        for key, value
        in signal_shape_features(
            values,
            float(
                sampling_frequency_hz
            ),
            int(
                nperseg
            ),
        ).items()
    }

    time = (
        np.arange(
            len(values),
            dtype=float,
        )
        / float(
            sampling_frequency_hz
        )
    )

    if len(values) >= 2:
        slope = float(
            np.polyfit(
                time,
                values,
                deg=1,
            )[
                0
            ]
        )
    else:
        slope = 0.0

    derivative = (
        np.diff(
            values
        )
        * float(
            sampling_frequency_hz
        )
    )

    features.update(
        {
            "median":
                float(
                    np.median(
                        values
                    )
                ),
            "iqr":
                float(
                    np.quantile(
                        values,
                        0.75,
                    )
                    - np.quantile(
                        values,
                        0.25,
                    )
                ),
            "range":
                float(
                    np.max(
                        values
                    )
                    - np.min(
                        values
                    )
                ),
            "linear_slope_per_second":
                slope,
            "derivative_mean":
                (
                    float(
                        np.mean(
                            derivative
                        )
                    )
                    if len(
                        derivative
                    )
                    else 0.0
                ),
            "derivative_std":
                (
                    float(
                        np.std(
                            derivative
                        )
                    )
                    if len(
                        derivative
                    )
                    else 0.0
                ),
            "derivative_mean_absolute":
                (
                    float(
                        np.mean(
                            np.abs(
                                derivative
                            )
                        )
                    )
                    if len(
                        derivative
                    )
                    else 0.0
                ),
        }
    )

    frequencies, spectrum = welch(
        values
        - np.mean(
            values
        ),
        fs=float(
            sampling_frequency_hz
        ),
        nperseg=min(
            int(
                nperseg
            ),
            len(
                values
            ),
        ),
    )

    for band_name, bounds in bands_hz.items():
        lowcut, highcut = [
            float(value)
            for value in bounds
        ]

        mask = (
            frequencies >= lowcut
        ) & (
            frequencies < highcut
        )

        if mask.sum() >= 2:
            power = float(
                np.trapz(
                    spectrum[
                        mask
                    ],
                    frequencies[
                        mask
                    ],
                )
            )
        elif mask.sum() == 1:
            power = float(
                spectrum[
                    mask
                ][
                    0
                ]
            )
        else:
            power = 0.0

        features[
            f"bandpower_{band_name}"
        ] = power

    if not all(
        np.isfinite(
            float(value)
        )
        for value in features.values()
    ):
        raise RuntimeError(
            "Pupil feature extraction produced nonfinite values."
        )

    return features


def prefix_features(
    modality: str,
    features: dict[str, float],
) -> dict[str, float]:
    """Prefix feature names to prevent multimodal collisions."""

    prefix = str(
        modality
    )

    return {
        f"{prefix}__{str(key)}":
            float(value)
        for key, value
        in features.items()
    }


def read_headerless_tsv_gz(
    archive: zipfile.ZipFile,
    member_name: str,
    column_name: str,
) -> np.ndarray:
    """Read one headerless compressed numerical signal."""

    compressed = archive.read(
        member_name
    )

    with gzip.GzipFile(
        fileobj=io.BytesIO(
            compressed
        ),
        mode="rb",
    ) as stream:
        frame = pd.read_csv(
            stream,
            sep="\t",
            header=None,
            names=[
                column_name
            ],
            low_memory=False,
        )

    return pd.to_numeric(
        frame[
            column_name
        ],
        errors="coerce",
    ).to_numpy(
        dtype=float
    )


def read_event_end_seconds(
    archive: zipfile.ZipFile,
    events_member: str,
) -> float:
    """Return the declared recording end time."""

    frame = pd.read_csv(
        io.BytesIO(
            archive.read(
                events_member
            )
        ),
        sep="\t",
        low_memory=False,
    )

    if frame.empty or "onset" not in frame.columns:
        raise ValueError(
            f"Invalid event table: {events_member}"
        )

    onset = pd.to_numeric(
        frame[
            "onset"
        ],
        errors="coerce",
    )

    if "event" in frame.columns:
        end_rows = frame[
            "event"
        ].astype(str).str.lower() == "end"

        end_values = onset.loc[
            end_rows
        ].dropna()

        if not end_values.empty:
            return float(
                end_values.max()
            )

    finite = onset.dropna()

    if finite.empty:
        raise ValueError(
            f"No finite event onset in {events_member}"
        )

    return float(
        finite.max()
    )


def derivative_eeg_member(
    raw_eeg_member: str,
) -> str:
    """Map one raw EEG member to its dataset derivative."""

    raw = str(
        raw_eeg_member
    )

    if not raw.endswith(
        "_eeg.bdf"
    ):
        raise ValueError(
            f"Unexpected EEG path: {raw}"
        )

    return (
        "derivatives/"
        + raw.replace(
            "_eeg.bdf",
            "_desc-eeg.bdf",
        )
    )


def select_manifest_record(
    records: pd.DataFrame,
    request: RecordingRequest,
) -> pd.Series:
    """Select exactly one primary manifest record."""

    primary = records.loc[
        robust_boolean_mask(
            records[
                "primary_cohort"
            ]
        )
    ].copy()

    selected = primary.loc[
        (
            primary[
                "subject"
            ].astype(str)
            == request.subject
        )
        & (
            primary[
                "session"
            ].astype(str)
            == request.session
        )
        & (
            primary[
                "task"
            ].astype(str)
            == request.task
        )
    ]

    if len(
        selected
    ) != 1:
        raise RuntimeError(
            f"Expected one record for {request}, "
            f"observed {len(selected)}."
        )

    return selected.iloc[
        0
    ]


def extract_recording(
    archive: zipfile.ZipFile,
    record: pd.Series,
    *,
    dataset_name: str,
    config: dict[str, Any],
    temporary_directory: Path,
) -> tuple[
    dict[str, list[dict[str, Any]]],
    dict[str, Any],
]:
    """Extract all aligned multimodal windows from one recording."""

    signal_config = config[
        "signals"
    ]

    window_config = config[
        "windowing"
    ]

    feature_config = config[
        "features"
    ]

    sampling_frequency_hz = float(
        signal_config[
            "sampling_frequency_hz"
        ]
    )

    eeg_channels = [
        str(value)
        for value in signal_config[
            "eeg"
        ][
            "channels"
        ]
    ]

    raw_eeg_member = str(
        record[
            "eeg_path"
        ]
    )

    eeg_member = derivative_eeg_member(
        raw_eeg_member
    )

    ecg_member = str(
        record[
            "ecg_path"
        ]
    )

    pupil_member = str(
        record[
            "pupil_path"
        ]
    )

    events_member = str(
        record[
            "events_path"
        ]
    )

    required_members = [
        eeg_member,
        ecg_member,
        pupil_member,
        events_member,
    ]

    archive_names = set(
        archive.namelist()
    )

    missing = [
        member
        for member in required_members
        if member not in archive_names
    ]

    if missing:
        raise FileNotFoundError(
            f"Required archive members missing: {missing}"
        )

    extracted_bdf = (
        temporary_directory
        / (
            dataset_name
            + "_"
            + Path(
                eeg_member
            ).name
        )
    )

    with archive.open(
        eeg_member,
        mode="r",
    ) as source:
        with extracted_bdf.open(
            "wb"
        ) as destination:
            while True:
                block = source.read(
                    1024
                    * 1024
                )

                if not block:
                    break

                destination.write(
                    block
                )

    raw = mne.io.read_raw_bdf(
        extracted_bdf,
        preload=True,
        verbose="ERROR",
    )

    if not np.isclose(
        float(
            raw.info[
                "sfreq"
            ]
        ),
        sampling_frequency_hz,
        atol=1e-9,
        rtol=0.0,
    ):
        raise RuntimeError(
            "Unexpected EEG sampling frequency: "
            f"{raw.info['sfreq']}"
        )

    missing_eeg_channels = [
        channel
        for channel in eeg_channels
        if channel not in raw.ch_names
    ]

    if missing_eeg_channels:
        raise RuntimeError(
            "Required EEG channels missing: "
            f"{missing_eeg_channels}"
        )

    eeg = raw.get_data(
        picks=eeg_channels
    ).T

    if bool(
        signal_config[
            "eeg"
        ][
            "convert_volts_to_microvolts"
        ]
    ):
        eeg = (
            eeg
            * 1_000_000.0
        )

    raw.close()

    try:
        extracted_bdf.unlink()
    except OSError:
        pass

    ecg_raw = read_headerless_tsv_gz(
        archive,
        ecg_member,
        "rawECG",
    )

    pupil_raw = read_headerless_tsv_gz(
        archive,
        pupil_member,
        "pupil_size",
    )

    ecg = preprocess_ecg(
        ecg_raw,
        sampling_frequency_hz=
            sampling_frequency_hz,
        bandpass_hz=
            signal_config[
                "ecg"
            ][
                "bandpass_hz"
            ],
        filter_order=int(
            signal_config[
                "ecg"
            ][
                "filter_order"
            ]
        ),
        minimum_finite_fraction=float(
            signal_config[
                "ecg"
            ][
                "minimum_finite_fraction"
            ]
        ),
    )

    maximum_gap_samples = int(
        round(
            float(
                signal_config[
                    "pupil"
                ][
                    "maximum_interpolation_gap_seconds"
                ]
            )
            * sampling_frequency_hz
        )
    )

    pupil, original_pupil_valid = (
        preprocess_pupil(
            pupil_raw,
            minimum_valid_value_exclusive=
                float(
                    signal_config[
                        "pupil"
                    ][
                        "minimum_valid_value_exclusive"
                    ]
                ),
            maximum_gap_samples=
                maximum_gap_samples,
        )
    )

    event_end_seconds = (
        read_event_end_seconds(
            archive,
            events_member,
        )
    )

    available_seconds = min(
        float(
            len(
                eeg
            )
            / sampling_frequency_hz
        ),
        float(
            len(
                ecg
            )
            / sampling_frequency_hz
        ),
        float(
            len(
                pupil
            )
            / sampling_frequency_hz
        ),
        event_end_seconds,
    )

    edge_trim_seconds = float(
        window_config[
            "edge_trim_seconds"
        ]
    )

    start_sample = int(
        np.ceil(
            edge_trim_seconds
            * sampling_frequency_hz
        )
    )

    stop_sample = int(
        np.floor(
            (
                available_seconds
                - edge_trim_seconds
            )
            * sampling_frequency_hz
        )
    )

    window_samples = int(
        round(
            float(
                window_config[
                    "duration_seconds"
                ]
            )
            * sampling_frequency_hz
        )
    )

    overlap_fraction = float(
        window_config[
            "overlap_fraction"
        ]
    )

    if not (
        0.0
        <= overlap_fraction
        < 1.0
    ):
        raise ValueError(
            "Window overlap fraction must be in [0, 1)."
        )

    step_samples = int(
        round(
            window_samples
            * (
                1.0
                - overlap_fraction
            )
        )
    )

    starts = window_start_indices(
        start_sample,
        stop_sample,
        window_samples,
        step_samples,
    )

    subject = str(
        record[
            "subject"
        ]
    )

    session = str(
        record[
            "session"
        ]
    )

    task = str(
        record[
            "task"
        ]
    )

    label = int(
        record[
            "label"
        ]
    )

    expected_label = int(
        config[
            "study"
        ][
            "condition_labels"
        ][
            session
        ]
    )

    if label != expected_label:
        raise RuntimeError(
            "Manifest label differs from the locked session mapping."
        )

    condition = str(
        config[
            "study"
        ][
            "condition_names"
        ][
            label
        ]
    )

    participant = namespaced_participant(
        dataset_name,
        subject,
    )

    recording_id = (
        f"{dataset_name}::{subject}::"
        f"{session}::{task}"
    )

    output = {
        table_name: []
        for table_name in MODALITY_TABLES
    }

    accepted = 0
    rejected_pupil = 0
    rejected_nonfinite_features = 0

    minimum_pupil_valid_fraction = float(
        signal_config[
            "pupil"
        ][
            "minimum_valid_fraction_per_window"
        ]
    )

    nperseg = int(
        feature_config[
            "welch_nperseg"
        ]
    )

    for window_index, start in enumerate(
        starts
    ):
        stop = (
            start
            + window_samples
        )

        pupil_valid_fraction = float(
            np.mean(
                original_pupil_valid[
                    start:stop
                ]
            )
        )

        pupil_window = pupil[
            start:stop
        ]

        if (
            pupil_valid_fraction
            < minimum_pupil_valid_fraction
            or not np.isfinite(
                pupil_window
            ).all()
        ):
            rejected_pupil += 1
            continue

        eeg_window = eeg[
            start:stop,
            :
        ]

        ecg_window = ecg[
            start:stop
        ]

        eeg_feature_values = prefix_features(
            "EEG",
            extract_eeg_features(
                eeg_window,
                sampling_frequency_hz,
                eeg_channels,
                feature_config[
                    "eeg_bands_hz"
                ],
                nperseg,
            ),
        )

        ecg_feature_values = prefix_features(
            "ECG",
            extract_ecg_features(
                ecg_window,
                sampling_frequency_hz,
                nperseg,
            ),
        )

        pupil_feature_values = prefix_features(
            "Pupil",
            pupil_features(
                pupil_window,
                sampling_frequency_hz=
                    sampling_frequency_hz,
                bands_hz=
                    feature_config[
                        "pupil_bands_hz"
                    ],
                nperseg=
                    nperseg,
            ),
        )

        combined_feature_values = {
            **eeg_feature_values,
            **ecg_feature_values,
            **pupil_feature_values,
        }

        if not all(
            np.isfinite(
                float(value)
            )
            for value
            in combined_feature_values.values()
        ):
            rejected_nonfinite_features += 1
            continue

        start_seconds = float(
            start
            / sampling_frequency_hz
        )

        end_seconds = float(
            stop
            / sampling_frequency_hz
        )

        segment_id = (
            f"{recording_id}::w"
            f"{window_index:05d}"
        )

        metadata = {
            "segment_id":
                segment_id,
            "recording_id":
                recording_id,
            "dataset":
                dataset_name,
            "participant":
                participant,
            "subject":
                subject,
            "session":
                session,
            "task":
                task,
            "label":
                label,
            "condition":
                condition,
            "window_index":
                int(
                    window_index
                ),
            "start_seconds":
                start_seconds,
            "end_seconds":
                end_seconds,
            "pupil_valid_fraction":
                pupil_valid_fraction,
        }

        output[
            "EEG"
        ].append(
            {
                **metadata,
                **eeg_feature_values,
            }
        )

        output[
            "ECG"
        ].append(
            {
                **metadata,
                **ecg_feature_values,
            }
        )

        output[
            "Pupil"
        ].append(
            {
                **metadata,
                **pupil_feature_values,
            }
        )

        output[
            "ECG_EEG"
        ].append(
            {
                **metadata,
                **ecg_feature_values,
                **eeg_feature_values,
            }
        )

        output[
            "ECG_Pupil"
        ].append(
            {
                **metadata,
                **ecg_feature_values,
                **pupil_feature_values,
            }
        )

        output[
            "EEG_Pupil"
        ].append(
            {
                **metadata,
                **eeg_feature_values,
                **pupil_feature_values,
            }
        )

        output[
            "ECG_EEG_Pupil"
        ].append(
            {
                **metadata,
                **ecg_feature_values,
                **eeg_feature_values,
                **pupil_feature_values,
            }
        )

        accepted += 1

    audit = {
        "recording_id":
            recording_id,
        "dataset":
            dataset_name,
        "participant":
            participant,
        "subject":
            subject,
        "session":
            session,
        "task":
            task,
        "label":
            label,
        "condition":
            condition,
        "event_end_seconds":
            event_end_seconds,
        "available_seconds":
            available_seconds,
        "candidate_windows":
            len(
                starts
            ),
        "accepted_windows":
            accepted,
        "rejected_pupil_windows":
            rejected_pupil,
        "rejected_nonfinite_feature_windows":
            rejected_nonfinite_features,
        "eeg_samples":
            int(
                len(
                    eeg
                )
            ),
        "ecg_samples":
            int(
                len(
                    ecg
                )
            ),
        "pupil_samples":
            int(
                len(
                    pupil
                )
            ),
        "original_pupil_valid_fraction":
            float(
                np.mean(
                    original_pupil_valid
                )
            ),
    }

    return output, audit


def validate_feature_tables(
    tables: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    """Validate multimodal alignment and finite schemas."""

    if list(
        tables
    ) != MODALITY_TABLES:
        raise RuntimeError(
            "Unexpected modality-table order."
        )

    reference = tables[
        "ECG_EEG_Pupil"
    ][
        METADATA_COLUMNS
    ].copy()

    if reference.empty:
        raise RuntimeError(
            "No accepted BBBD windows were produced."
        )

    if reference[
        "segment_id"
    ].duplicated().any():
        raise RuntimeError(
            "Duplicate BBBD segment identifiers detected."
        )

    feature_counts = {}

    for table_name, frame in tables.items():
        if len(
            frame
        ) != len(
            reference
        ):
            raise RuntimeError(
                f"{table_name}: row count differs from the fused table."
            )

        for column in METADATA_COLUMNS:
            if not frame[
                column
            ].astype(str).equals(
                reference[
                    column
                ].astype(str)
            ):
                raise RuntimeError(
                    f"{table_name}: metadata alignment changed for {column}."
                )

        feature_columns = [
            column
            for column in frame.columns
            if column not in METADATA_COLUMNS
        ]

        if not feature_columns:
            raise RuntimeError(
                f"{table_name}: no feature columns."
            )

        values = frame[
            feature_columns
        ].to_numpy(
            dtype=float
        )

        if not np.isfinite(
            values
        ).all():
            raise RuntimeError(
                f"{table_name}: nonfinite feature values detected."
            )

        feature_counts[
            table_name
        ] = len(
            feature_columns
        )

    return {
        "rows":
            int(
                len(
                    reference
                )
            ),
        "participants":
            int(
                reference[
                    "participant"
                ].nunique()
            ),
        "recordings":
            int(
                reference[
                    "recording_id"
                ].nunique()
            ),
        "label_counts": {
            str(
                int(
                    key
                )
            ):
                int(
                    value
                )
            for key, value
            in reference[
                "label"
            ]
            .value_counts()
            .sort_index()
            .to_dict()
            .items()
        },
        "feature_counts":
            feature_counts,
    }


def requested_smoke_records(
    config: dict[str, Any],
) -> list[RecordingRequest]:
    """Return the prespecified smoke records."""

    return [
        RecordingRequest(
            dataset=str(
                item[
                    "dataset"
                ]
            ),
            subject=str(
                item[
                    "subject"
                ]
            ),
            session=str(
                item[
                    "session"
                ]
            ),
            task=str(
                item[
                    "task"
                ]
            ),
        )
        for item in config[
            "smoke_records"
        ]
    ]


def build_bbbd_features(
    *,
    config_path: str | Path,
    output_directory: str | Path,
    smoke: bool,
) -> dict[str, Any]:
    """Build aligned BBBD seven-path feature tables."""

    config = load_bbbd_config(
        config_path
    )

    project_config = load_revision_config(
        check_input_paths=False
    )

    output_root = resolve_project_path(
        output_directory
    )

    if output_root.exists():
        raise FileExistsError(
            f"BBBD output already exists: {output_root}"
        )

    output_root.mkdir(
        parents=True,
        exist_ok=False,
    )

    direct_cohort_participants = (
        load_direct_cohort_participants(
            config
        )
    )

    if smoke:
        requests = requested_smoke_records(
            config
        )

        outside_cohort = [
            request
            for request in requests
            if namespaced_participant(
                request.dataset,
                request.subject,
            )
            not in direct_cohort_participants
        ]

        if outside_cohort:
            raise RuntimeError(
                "Smoke records fall outside the locked "
                f"complete-case cohort: {outside_cohort}"
            )

    else:
        requests = []

        for dataset_name, dataset_config in config[
            "datasets"
        ].items():
            manifest = pd.read_csv(
                resolve_project_path(
                    dataset_config[
                        "manifest"
                    ]
                ),
                low_memory=False,
            )

            primary = manifest.loc[
                robust_boolean_mask(
                    manifest[
                        "primary_cohort"
                    ]
                )
            ].copy()

            primary[
                "_namespaced_participant"
            ] = [
                namespaced_participant(
                    str(
                        dataset_name
                    ),
                    str(
                        subject
                    ),
                )
                for subject in primary[
                    "subject"
                ].astype(str)
            ]

            primary = primary.loc[
                primary[
                    "_namespaced_participant"
                ].isin(
                    direct_cohort_participants
                )
            ].copy()

            for row in primary.itertuples(
                index=False
            ):
                requests.append(
                    RecordingRequest(
                        dataset=str(
                            dataset_name
                        ),
                        subject=str(
                            row.subject
                        ),
                        session=str(
                            row.session
                        ),
                        task=str(
                            row.task
                        ),
                    )
                )

        expected_recordings = int(
            config[
                "cohorts"
            ][
                "direct_seven_path"
            ][
                "expected_recordings"
            ]
        )

        if len(
            requests
        ) != expected_recordings:
            raise RuntimeError(
                "Full BBBD request count differs from the "
                f"locked complete-case cohort: {len(requests)} "
                f"versus {expected_recordings}."
            )

    rows_by_table = {
        table_name: []
        for table_name in MODALITY_TABLES
    }

    audit_rows = []

    manifests = {}

    archives = {}

    try:
        for dataset_name, dataset_config in config[
            "datasets"
        ].items():
            manifest_path = resolve_project_path(
                dataset_config[
                    "manifest"
                ]
            )

            manifests[
                dataset_name
            ] = pd.read_csv(
                manifest_path,
                low_memory=False,
            )

            archive_key = str(
                dataset_config[
                    "archive_config_key"
                ]
            )

            archive_path = Path(
                project_config[
                    "paths"
                ][
                    archive_key
                ]
            )

            if not archive_path.is_file():
                raise FileNotFoundError(
                    f"BBBD archive missing: {archive_path}"
                )

            archives[
                dataset_name
            ] = zipfile.ZipFile(
                archive_path,
                mode="r",
            )

        with tempfile.TemporaryDirectory() as temporary:
            temporary_directory = Path(
                temporary
            )

            for index, request in enumerate(
                requests,
                start=1,
            ):
                print(
                    f"[{index:03d}/{len(requests):03d}] "
                    f"{request.dataset} "
                    f"{request.subject} "
                    f"{request.session} "
                    f"{request.task}",
                    flush=True,
                )

                if request.dataset not in manifests:
                    raise KeyError(
                        f"Unknown BBBD dataset: {request.dataset}"
                    )

                record = select_manifest_record(
                    manifests[
                        request.dataset
                    ],
                    request,
                )

                extracted, audit = extract_recording(
                    archives[
                        request.dataset
                    ],
                    record,
                    dataset_name=
                        request.dataset,
                    config=
                        config,
                    temporary_directory=
                        temporary_directory,
                )

                for table_name in MODALITY_TABLES:
                    rows_by_table[
                        table_name
                    ].extend(
                        extracted[
                            table_name
                        ]
                    )

                audit_rows.append(
                    audit
                )

    finally:
        for archive in archives.values():
            archive.close()

    tables = {
        table_name:
            pd.DataFrame(
                rows_by_table[
                    table_name
                ]
            )
        for table_name in MODALITY_TABLES
    }

    validation = validate_feature_tables(
        tables
    )

    for table_name, frame in tables.items():
        frame.to_csv(
            output_root
            / f"{table_name}_features.csv",
            index=False,
        )

    audit_frame = pd.DataFrame(
        audit_rows
    )

    audit_frame.to_csv(
        output_root
        / "recording_audit.csv",
        index=False,
    )

    summary = {
        "dataset":
            "BBBD",
        "target":
            config[
                "study"
            ][
                "target_name"
            ],
        "smoke_run":
            bool(
                smoke
            ),
        "cohort":
            config[
                "cohorts"
            ][
                "direct_seven_path"
            ][
                "name"
            ],
        "locked_cohort_participants":
            int(
                config[
                    "cohorts"
                ][
                    "direct_seven_path"
                ][
                    "expected_participants"
                ]
            ),
        "locked_cohort_recordings":
            int(
                config[
                    "cohorts"
                ][
                    "direct_seven_path"
                ][
                    "expected_recordings"
                ]
            ),
        "requested_recordings":
            len(
                requests
            ),
        "completed_recordings":
            int(
                len(
                    audit_frame
                )
            ),
        "candidate_windows":
            int(
                audit_frame[
                    "candidate_windows"
                ].sum()
            ),
        "accepted_windows":
            int(
                audit_frame[
                    "accepted_windows"
                ].sum()
            ),
        "rejected_pupil_windows":
            int(
                audit_frame[
                    "rejected_pupil_windows"
                ].sum()
            ),
        "rejected_nonfinite_feature_windows":
            int(
                audit_frame[
                    "rejected_nonfinite_feature_windows"
                ].sum()
            ),
        "window_duration_seconds":
            float(
                config[
                    "windowing"
                ][
                    "duration_seconds"
                ]
            ),
        "window_overlap_fraction":
            float(
                config[
                    "windowing"
                ][
                    "overlap_fraction"
                ]
            ),
        "sampling_frequency_hz":
            float(
                config[
                    "signals"
                ][
                    "sampling_frequency_hz"
                ]
            ),
        "eeg_source":
            config[
                "signals"
            ][
                "eeg"
            ][
                "source"
            ],
        "ecg_source":
            config[
                "signals"
            ][
                "ecg"
            ][
                "source"
            ],
        "pupil_source":
            config[
                "signals"
            ][
                "pupil"
            ][
                "source"
            ],
        "validation":
            validation,
        "known_limitation":
            config[
                "study"
            ][
                "known_limitation"
            ],
        "classification_performed":
            False,
        "feature_selection_performed":
            False,
    }

    (
        output_root
        / "summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        "\n===== BBBD FEATURE SUMMARY ====="
    )

    print(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        )
    )

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Construct aligned BBBD multimodal feature tables."
        )
    )

    parser.add_argument(
        "--config",
        default="configs/bbbd.yaml",
    )

    parser.add_argument(
        "--output-directory",
        default=None,
    )

    parser.add_argument(
        "--smoke",
        action="store_true",
    )

    arguments = parser.parse_args()

    config = load_bbbd_config(
        arguments.config
    )

    if arguments.output_directory is None:
        output_directory = (
            config[
                "outputs"
            ][
                "smoke_directory"
                if arguments.smoke
                else "full_directory"
            ]
        )
    else:
        output_directory = (
            arguments.output_directory
        )

    build_bbbd_features(
        config_path=
            arguments.config,
        output_directory=
            output_directory,
        smoke=
            arguments.smoke,
    )


if __name__ == "__main__":
    main()