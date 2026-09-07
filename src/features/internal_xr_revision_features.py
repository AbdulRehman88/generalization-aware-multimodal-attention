"""Scientifically corrected feature extraction for internal XR signals.

The module deliberately excludes:

- EEG frequencies above the validated 40 Hz preprocessing cutoff.
- Four-second frequency-domain HRV estimates.
- Misnamed PDE-derived features.
- Absolute horizontal and vertical gaze-position means.
- Global feature normalization, selection, or cleanup.

Normalization, constant-feature removal, duplicate-feature removal, and SHAP
selection must be fitted using training data only during model evaluation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import kurtosis, skew

from src.core.config import load_revision_config


PROJECT_ROOT = Path(__file__).resolve().parents[2]

METADATA_COLUMNS = [
    "segment_id",
    "participant",
    "phase",
    "label",
    "window_index",
    "pupil_available",
]

FORBIDDEN_FEATURE_TOKENS = (
    "high_gamma",
    "pde",
    "wave_equation",
    "burgers",
    "diffusion",
    "telegraph",
    "klein_gordon",
    "hrv",
    "lf_hf",
)


def safe_skew(values: np.ndarray) -> float:
    """Return finite skewness, using zero for constant signals."""

    array = np.asarray(values, dtype=float)

    if np.std(array) <= 1e-12:
        return 0.0

    result = float(
        skew(
            array,
            bias=False,
            nan_policy="raise",
        )
    )

    return result if np.isfinite(result) else 0.0


def safe_kurtosis(values: np.ndarray) -> float:
    """Return finite excess kurtosis, using zero for constants."""

    array = np.asarray(values, dtype=float)

    if np.std(array) <= 1e-12:
        return 0.0

    result = float(
        kurtosis(
            array,
            fisher=True,
            bias=False,
            nan_policy="raise",
        )
    )

    return result if np.isfinite(result) else 0.0


def compute_psd(
    values: np.ndarray,
    sampling_rate_hz: float,
    nperseg: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute a finite one-sided Welch power spectral density."""

    array = np.asarray(values, dtype=float)

    frequencies, power = welch(
        array,
        fs=sampling_rate_hz,
        nperseg=min(len(array), int(nperseg)),
        detrend="constant",
        scaling="density",
    )

    power = np.maximum(
        np.asarray(power, dtype=float),
        0.0,
    )

    return frequencies, power


def integrate_band(
    frequencies: np.ndarray,
    power: np.ndarray,
    lower_hz: float,
    upper_hz: float,
) -> float:
    """Integrate PSD within a half-open frequency interval."""

    mask = (
        (frequencies >= lower_hz)
        & (frequencies < upper_hz)
    )

    selected_frequency = frequencies[mask]
    selected_power = power[mask]

    if len(selected_frequency) == 0:
        return 0.0

    if len(selected_frequency) == 1:
        if len(frequencies) > 1:
            resolution = float(
                np.median(np.diff(frequencies))
            )
        else:
            resolution = 1.0

        return float(
            selected_power[0] * resolution
        )

    return float(
        np.trapezoid(
            selected_power,
            selected_frequency,
        )
    )


def spectral_entropy(
    power: np.ndarray,
) -> float:
    """Compute normalized Shannon entropy of a PSD."""

    positive = np.asarray(
        power,
        dtype=float,
    )

    total = float(positive.sum())

    if total <= 1e-20 or len(positive) <= 1:
        return 0.0

    probabilities = positive / total
    probabilities = probabilities[
        probabilities > 0
    ]

    entropy = -float(
        np.sum(
            probabilities
            * np.log(probabilities)
        )
    )

    maximum = float(
        np.log(len(positive))
    )

    return entropy / maximum if maximum > 0 else 0.0


def signal_shape_features(
    values: np.ndarray,
    sampling_rate_hz: float,
    nperseg: int,
) -> dict[str, float]:
    """Extract finite time-domain and nonlinear shape descriptors."""

    array = np.asarray(
        values,
        dtype=float,
    ).reshape(-1)

    if len(array) < 4:
        raise ValueError(
            "At least four samples are required."
        )

    if not np.isfinite(array).all():
        raise ValueError(
            "Feature input contains non-finite values."
        )

    first_difference = np.diff(array)
    second_difference = np.diff(
        first_difference
    )

    variance_signal = float(
        np.var(array)
    )

    variance_first = float(
        np.var(first_difference)
    )

    variance_second = float(
        np.var(second_difference)
    )

    if variance_signal > 1e-20:
        mobility = float(
            np.sqrt(
                variance_first
                / variance_signal
            )
        )
    else:
        mobility = 0.0

    if (
        variance_first > 1e-20
        and mobility > 1e-20
    ):
        second_mobility = float(
            np.sqrt(
                variance_second
                / variance_first
            )
        )

        complexity = (
            second_mobility
            / mobility
        )
    else:
        complexity = 0.0

    centered = array - float(
        np.mean(array)
    )

    zero_crossings = float(
        np.mean(
            centered[:-1]
            * centered[1:]
            < 0
        )
    )

    frequencies, power = compute_psd(
        array,
        sampling_rate_hz,
        nperseg,
    )

    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "median": float(np.median(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "peak_to_peak": float(np.ptp(array)),
        "rms": float(
            np.sqrt(
                np.mean(array ** 2)
            )
        ),
        "skew": safe_skew(array),
        "kurtosis": safe_kurtosis(array),
        "mean_abs_difference": float(
            np.mean(
                np.abs(first_difference)
            )
        ),
        "zero_crossing_rate": zero_crossings,
        "hjorth_mobility": mobility,
        "hjorth_complexity": complexity,
        "spectral_entropy":
            spectral_entropy(power),
    }


def extract_eeg_features(
    window: np.ndarray,
    sampling_rate_hz: float,
    channels: list[str],
    bands: dict[str, list[float]],
    nperseg: int,
) -> dict[str, float]:
    """Extract corrected channel-wise EEG descriptors."""

    array = np.asarray(
        window,
        dtype=float,
    )

    if array.ndim != 2:
        raise ValueError(
            "EEG window must be samples by channels."
        )

    if array.shape[1] != len(channels):
        raise ValueError(
            "EEG channel count does not match schema."
        )

    features: dict[str, float] = {}

    for channel_index, channel in enumerate(channels):
        signal = array[:, channel_index]

        shape = signal_shape_features(
            signal,
            sampling_rate_hz,
            nperseg,
        )

        for name, value in shape.items():
            features[
                f"eeg_{channel}_{name}"
            ] = value

        frequencies, power = compute_psd(
            signal,
            sampling_rate_hz,
            nperseg,
        )

        total_power = integrate_band(
            frequencies,
            power,
            0.5,
            40.0,
        )

        features[
            f"eeg_{channel}_log_total_power_0p5_40"
        ] = float(
            np.log10(
                total_power + 1e-20
            )
        )

        absolute_powers: dict[str, float] = {}

        for band_name, limits in bands.items():
            lower_hz, upper_hz = map(
                float,
                limits,
            )

            band_power = integrate_band(
                frequencies,
                power,
                lower_hz,
                upper_hz,
            )

            absolute_powers[
                band_name
            ] = band_power

            features[
                f"eeg_{channel}_log_power_{band_name}"
            ] = float(
                np.log10(
                    band_power + 1e-20
                )
            )

            features[
                f"eeg_{channel}_relative_power_{band_name}"
            ] = float(
                band_power
                / (total_power + 1e-20)
            )

        theta = absolute_powers["theta"]
        alpha = absolute_powers["alpha"]
        beta = absolute_powers["beta"]

        features[
            f"eeg_{channel}_ratio_theta_alpha"
        ] = float(
            theta / (alpha + 1e-20)
        )

        features[
            f"eeg_{channel}_ratio_theta_beta"
        ] = float(
            theta / (beta + 1e-20)
        )

        features[
            f"eeg_{channel}_ratio_alpha_beta"
        ] = float(
            alpha / (beta + 1e-20)
        )

        features[
            f"eeg_{channel}_ratio_beta_theta"
        ] = float(
            beta / (theta + 1e-20)
        )

    return features


def extract_ecg_features(
    window: np.ndarray,
    sampling_rate_hz: float,
    nperseg: int,
) -> dict[str, float]:
    """Extract ECG waveform descriptors without invalid HRV claims."""

    signal = np.asarray(
        window,
        dtype=float,
    ).reshape(-1)

    shape = signal_shape_features(
        signal,
        sampling_rate_hz,
        nperseg,
    )

    features = {
        f"ecg_{name}": value
        for name, value in shape.items()
    }

    frequencies, power = compute_psd(
        signal,
        sampling_rate_hz,
        nperseg,
    )

    waveform_power = integrate_band(
        frequencies,
        power,
        0.5,
        40.0,
    )

    physiological_mask = (
        (frequencies >= 0.5)
        & (frequencies <= 4.0)
    )

    if physiological_mask.any():
        local_power = power[
            physiological_mask
        ]

        local_frequencies = frequencies[
            physiological_mask
        ]

        dominant_frequency = float(
            local_frequencies[
                int(np.argmax(local_power))
            ]
        )
    else:
        dominant_frequency = 0.0

    features[
        "ecg_waveform_log_power_0p5_40"
    ] = float(
        np.log10(
            waveform_power + 1e-20
        )
    )

    features[
        "ecg_waveform_dominant_frequency_0p5_4_hz"
    ] = dominant_frequency

    return features


def extract_pupil_features(
    window: np.ndarray,
    sampling_rate_hz: float,
    bands: dict[str, list[float]],
    nperseg: int,
) -> dict[str, float]:
    """Extract valid pupil-size and gaze-dynamics descriptors."""

    array = np.asarray(
        window,
        dtype=float,
    )

    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(
            "Pupil window must contain x, y, and radius."
        )

    if not np.isfinite(array).all():
        raise ValueError(
            "Pupil feature extraction received missing values."
        )

    x = array[:, 0]
    y = array[:, 1]
    radius = array[:, 2]

    features: dict[str, float] = {}

    radius_shape = signal_shape_features(
        radius,
        sampling_rate_hz,
        nperseg,
    )

    for name, value in radius_shape.items():
        features[
            f"pupil_radius_{name}"
        ] = value

    time = np.arange(
        len(radius),
        dtype=float,
    ) / sampling_rate_hz

    slope = float(
        np.polyfit(
            time,
            radius,
            deg=1,
        )[0]
    )

    radius_velocity = (
        np.diff(radius)
        * sampling_rate_hz
    )

    features[
        "pupil_radius_linear_slope"
    ] = slope

    features[
        "pupil_radius_velocity_mean_abs"
    ] = float(
        np.mean(
            np.abs(radius_velocity)
        )
    )

    features[
        "pupil_radius_velocity_std"
    ] = float(
        np.std(radius_velocity)
    )

    features[
        "pupil_radius_velocity_max_abs"
    ] = float(
        np.max(
            np.abs(radius_velocity)
        )
    )

    frequencies, power = compute_psd(
        radius,
        sampling_rate_hz,
        nperseg,
    )

    total_power = integrate_band(
        frequencies,
        power,
        0.25,
        4.0,
    )

    features[
        "pupil_radius_log_power_0p25_4"
    ] = float(
        np.log10(
            total_power + 1e-20
        )
    )

    for band_name, limits in bands.items():
        lower_hz, upper_hz = map(
            float,
            limits,
        )

        band_power = integrate_band(
            frequencies,
            power,
            lower_hz,
            upper_hz,
        )

        features[
            f"pupil_radius_relative_power_{band_name}"
        ] = float(
            band_power
            / (total_power + 1e-20)
        )

    for axis_name, axis_values in (
        ("x", x),
        ("y", y),
    ):
        centered = (
            axis_values
            - np.mean(axis_values)
        )

        difference = np.diff(
            centered
        )

        velocity = (
            difference
            * sampling_rate_hz
        )

        features[
            f"gaze_{axis_name}_std"
        ] = float(
            np.std(centered)
        )

        features[
            f"gaze_{axis_name}_peak_to_peak"
        ] = float(
            np.ptp(centered)
        )

        features[
            f"gaze_{axis_name}_mean_abs_step"
        ] = float(
            np.mean(
                np.abs(difference)
            )
        )

        features[
            f"gaze_{axis_name}_velocity_std"
        ] = float(
            np.std(velocity)
        )

        features[
            f"gaze_{axis_name}_velocity_max_abs"
        ] = float(
            np.max(
                np.abs(velocity)
            )
        )

    dx = np.diff(x)
    dy = np.diff(y)

    path_step = np.sqrt(
        dx ** 2 + dy ** 2
    )

    features[
        "gaze_path_mean_step"
    ] = float(
        np.mean(path_step)
    )

    features[
        "gaze_path_std_step"
    ] = float(
        np.std(path_step)
    )

    features[
        "gaze_path_max_step"
    ] = float(
        np.max(path_step)
    )

    return features


def resolve_directory(
    configured_path: str,
) -> Path:
    """Resolve a project-relative or absolute directory."""

    path = Path(configured_path)

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path.resolve()


def feature_columns(
    frame: pd.DataFrame,
) -> list[str]:
    """Return feature-only columns."""

    return [
        column
        for column in frame.columns
        if column not in METADATA_COLUMNS
    ]


def validate_feature_table(
    frame: pd.DataFrame,
    table_name: str,
) -> None:
    """Validate schema, identifiers, and finite numerical values."""

    missing_metadata = [
        column
        for column in METADATA_COLUMNS
        if column not in frame.columns
    ]

    if missing_metadata:
        raise RuntimeError(
            f"{table_name} is missing metadata columns: "
            f"{missing_metadata}"
        )

    if frame["segment_id"].duplicated().any():
        raise RuntimeError(
            f"{table_name} contains duplicate segment identifiers."
        )

    columns = feature_columns(frame)

    if not columns:
        raise RuntimeError(
            f"{table_name} contains no features."
        )

    forbidden = [
        column
        for column in columns
        if any(
            token in column.casefold()
            for token in FORBIDDEN_FEATURE_TOKENS
        )
    ]

    if forbidden:
        raise RuntimeError(
            f"{table_name} contains forbidden features: "
            f"{forbidden[:10]}"
        )

    values = frame[
        columns
    ].to_numpy(dtype=float)

    if not np.isfinite(values).all():
        bad_count = int(
            (~np.isfinite(values)).sum()
        )

        raise RuntimeError(
            f"{table_name} contains "
            f"{bad_count} non-finite feature values."
        )


def fuse_tables(
    left: pd.DataFrame,
    right: pd.DataFrame,
) -> pd.DataFrame:
    """Fuse two feature tables using exact segment identifiers."""

    left_features = feature_columns(left)
    right_features = feature_columns(right)

    overlap = set(
        left_features
    ).intersection(
        right_features
    )

    if overlap:
        raise RuntimeError(
            f"Feature-name collision during fusion: "
            f"{sorted(overlap)[:10]}"
        )

    fused = left[
        METADATA_COLUMNS
        + left_features
    ].merge(
        right[
            ["segment_id"]
            + right_features
        ],
        on="segment_id",
        how="inner",
        validate="one_to_one",
    )

    return fused


def build_internal_xr_features(
    config: dict[str, Any],
    *,
    write_outputs: bool,
    explicit_input: str | None = None,
    explicit_output: str | None = None,
) -> dict[str, Any]:
    """Extract corrected internal XR feature matrices."""

    if explicit_input is None:
        input_directory = resolve_directory(
            config[
                "datasets"
            ][
                "internal_xr"
            ][
                "output_dir"
            ]
        )
    else:
        input_directory = Path(
            explicit_input
        ).resolve()

    if explicit_output is None:
        output_directory = resolve_directory(
            config[
                "features"
            ][
                "output_dir"
            ]
        )
    else:
        output_directory = Path(
            explicit_output
        ).resolve()

    manifest_path = (
        input_directory
        / "window_manifest.csv"
    )

    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Preprocessing manifest not found: {manifest_path}"
        )

    manifest = pd.read_csv(
        manifest_path,
        low_memory=False,
    )

    expected_metadata = [
        "segment_id",
        "participant",
        "phase",
        "label",
        "window_index",
        "pupil_available",
        "recording_archive",
        "archive_window_index",
    ]

    missing = [
        column
        for column in expected_metadata
        if column not in manifest.columns
    ]

    if missing:
        raise RuntimeError(
            f"Preprocessing manifest is missing: {missing}"
        )

    if manifest["segment_id"].duplicated().any():
        raise RuntimeError(
            "Preprocessing manifest contains duplicate segment IDs."
        )

    feature_config = config["features"]

    nperseg = int(
        feature_config.get(
            "welch_nperseg",
            256,
        )
    )

    eeg_bands = feature_config[
        "eeg_bands_hz"
    ]

    pupil_bands = feature_config[
        "pupil_bands_hz"
    ]

    eeg_channels = list(
        config["eeg"]["internal_channels"]
    )

    eeg_rows: list[dict[str, Any]] = []
    ecg_rows: list[dict[str, Any]] = []
    pupil_rows: list[dict[str, Any]] = []

    grouped = manifest.groupby(
        "recording_archive",
        sort=True,
    )

    recording_count = grouped.ngroups

    for recording_number, (
        relative_archive,
        recording_manifest,
    ) in enumerate(
        grouped,
        start=1,
    ):
        archive_path = (
            input_directory
            / str(relative_archive)
        )

        if not archive_path.is_file():
            raise FileNotFoundError(
                f"Recording archive not found: {archive_path}"
            )

        with np.load(
            archive_path,
            allow_pickle=False,
        ) as archive:
            eeg = archive["eeg"]
            ecg = archive["ecg"]
            pupil = archive["pupil"]

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

            archive_channels = [
                str(value)
                for value in archive[
                    "eeg_channels"
                ].tolist()
            ]

            if archive_channels != eeg_channels:
                raise RuntimeError(
                    f"EEG channel schema mismatch in {archive_path}: "
                    f"{archive_channels}"
                )

            recording_manifest = (
                recording_manifest
                .sort_values(
                    "archive_window_index"
                )
            )

            for row in recording_manifest.itertuples(
                index=False
            ):
                window_index = int(
                    row.archive_window_index
                )

                metadata = {
                    "segment_id":
                        str(row.segment_id),
                    "participant":
                        str(row.participant),
                    "phase":
                        int(row.phase),
                    "label":
                        int(row.label),
                    "window_index":
                        int(row.window_index),
                    "pupil_available":
                        bool(row.pupil_available),
                }

                eeg_features = extract_eeg_features(
                    eeg[window_index],
                    eeg_sampling_rate,
                    eeg_channels,
                    eeg_bands,
                    nperseg,
                )

                ecg_features = extract_ecg_features(
                    ecg[window_index],
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

                if bool(row.pupil_available):
                    pupil_features = extract_pupil_features(
                        pupil[window_index],
                        pupil_sampling_rate,
                        pupil_bands,
                        nperseg,
                    )

                    pupil_rows.append(
                        {
                            **metadata,
                            **pupil_features,
                        }
                    )

        if (
            recording_number % 5 == 0
            or recording_number == recording_count
        ):
            print(
                f"Processed recordings: "
                f"{recording_number}/{recording_count}",
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

    validate_feature_table(
        eeg_table,
        "EEG",
    )

    validate_feature_table(
        ecg_table,
        "ECG",
    )

    validate_feature_table(
        pupil_table,
        "Pupil",
    )

    ecg_eeg = fuse_tables(
        ecg_table,
        eeg_table,
    )

    ecg_pupil = fuse_tables(
        ecg_table,
        pupil_table,
    )

    eeg_pupil = fuse_tables(
        eeg_table,
        pupil_table,
    )

    trimodal = fuse_tables(
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

    for table_name, table in tables.items():
        validate_feature_table(
            table,
            table_name,
        )

    if len(eeg_table) != len(manifest):
        raise RuntimeError(
            "EEG feature row count does not match preprocessing manifest."
        )

    if len(ecg_table) != len(manifest):
        raise RuntimeError(
            "ECG feature row count does not match preprocessing manifest."
        )

    expected_pupil_rows = int(
        manifest[
            "pupil_available"
        ].sum()
    )

    if len(pupil_table) != expected_pupil_rows:
        raise RuntimeError(
            "Pupil feature row count does not match quality manifest."
        )

    summary = {
        "dataset": "internal_xr_attention",
        "source_manifest": str(
            manifest_path
        ),
        "feature_profile": "corrected",
        "normalization_applied": False,
        "feature_selection_applied": False,
        "constant_feature_removal_applied": False,
        "duplicate_feature_removal_applied": False,
        "cleanup_policy":
            "Fit using training data only.",
        "excluded_feature_families": [
            "EEG power above 40 Hz",
            "four-second frequency-domain HRV",
            "legacy PDE-named gradient features",
            "absolute gaze x and y means",
        ],
        "eeg_channels": eeg_channels,
        "eeg_bands_hz": eeg_bands,
        "pupil_bands_hz": pupil_bands,
        "tables": {
            table_name: {
                "rows": int(
                    len(table)
                ),
                "features": int(
                    len(
                        feature_columns(
                            table
                        )
                    )
                ),
                "label_counts": {
                    str(int(label)): int(count)
                    for label, count in (
                        table[
                            "label"
                        ]
                        .value_counts()
                        .sort_index()
                        .items()
                    )
                },
            }
            for table_name, table in tables.items()
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
                "Refusing to overwrite existing feature output: "
                f"{output_directory}"
            )

        if temporary_directory.exists():
            raise FileExistsError(
                "Temporary feature-output directory already exists: "
                f"{temporary_directory}"
            )

        temporary_directory.mkdir(
            parents=True,
            exist_ok=False,
        )

        for table_name, table in tables.items():
            table.to_csv(
                temporary_directory
                / f"{table_name}_features.csv",
                index=False,
            )

        (
            temporary_directory
            / "feature_summary.json"
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
            / "feature_schema.json"
        ).write_text(
            json.dumps(
                {
                    table_name:
                        feature_columns(table)
                    for table_name, table
                    in tables.items()
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        temporary_directory.replace(
            output_directory
        )

    print("\n===== CORRECTED FEATURE SUMMARY =====")

    for table_name, table_summary in (
        summary["tables"].items()
    ):
        print(
            f"{table_name:18s} "
            f"rows={table_summary['rows']:4d}, "
            f"features={table_summary['features']:3d}, "
            f"labels={table_summary['label_counts']}"
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
            "Extract corrected internal XR features."
        )
    )

    parser.add_argument(
        "--write",
        action="store_true",
        help="Write corrected feature matrices.",
    )

    parser.add_argument(
        "--input-root",
        default=None,
        help="Optional corrected preprocessing directory.",
    )

    parser.add_argument(
        "--output-root",
        default=None,
        help="Optional corrected feature output directory.",
    )

    arguments = parser.parse_args()

    config = load_revision_config(
        check_input_paths=True
    )

    build_internal_xr_features(
        config,
        write_outputs=arguments.write,
        explicit_input=arguments.input_root,
        explicit_output=arguments.output_root,
    )


if __name__ == "__main__":
    main()