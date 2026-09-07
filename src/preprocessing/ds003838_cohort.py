"""Resumable cohort preprocessing for OpenNeuro ds003838."""

from __future__ import annotations

import argparse
import json
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.core.config import load_revision_config
from src.preprocessing.ds003838_revision import (
    PROJECT_ROOT,
    preprocess_subject,
    write_subject_output,
)


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path.resolve()


def boolean_series(series: pd.Series) -> pd.Series:
    """Convert manifest Boolean values without treating 'False' as true."""

    if series.dtype == bool:
        return series.astype(bool)

    normalized = (
        series.astype(str)
        .str.strip()
        .str.lower()
    )

    mapped = normalized.map(
        {
            "true": True,
            "1": True,
            "yes": True,
            "false": False,
            "0": False,
            "no": False,
            "nan": False,
            "": False,
        }
    )

    if mapped.isna().any():
        unknown = sorted(
            normalized.loc[
                mapped.isna()
            ].unique().tolist()
        )

        raise ValueError(
            f"Unrecognized Boolean values: {unknown}"
        )

    return mapped.astype(bool)


def verify_subject_output(
    subject_directory: Path,
    *,
    expected_subject: str,
    expected_channels: int,
    expected_trials: int,
) -> dict[str, Any]:
    """Verify one completed participant before using or skipping it."""

    required = [
        subject_directory / "windows.npz",
        subject_directory / "metadata.csv",
        subject_directory / "pupil_quality.csv",
        subject_directory / "summary.json",
    ]

    missing = [
        str(path)
        for path in required
        if not path.is_file()
    ]

    if missing:
        raise RuntimeError(
            f"Incomplete participant output for {expected_subject}: "
            f"{missing}"
        )

    summary = json.loads(
        (
            subject_directory / "summary.json"
        ).read_text(encoding="utf-8")
    )

    metadata = pd.read_csv(
        subject_directory / "metadata.csv",
        low_memory=False,
    )

    quality = pd.read_csv(
        subject_directory / "pupil_quality.csv",
        low_memory=False,
    )

    with np.load(
        subject_directory / "windows.npz",
        allow_pickle=False,
    ) as archive:
        eeg_shape = tuple(archive["eeg"].shape)
        ecg_shape = tuple(archive["ecg"].shape)
        pupil_shape = tuple(archive["pupil"].shape)

        eeg_finite = bool(
            np.isfinite(archive["eeg"]).all()
        )

        ecg_finite = bool(
            np.isfinite(archive["ecg"]).all()
        )

        pupil = archive["pupil"]

    if str(summary["subject"]) != expected_subject:
        raise RuntimeError(
            f"Subject mismatch in {subject_directory}."
        )

    expected_eeg_shape = (
        expected_trials,
        512,
        expected_channels,
    )

    expected_ecg_shape = (
        expected_trials,
        512,
        1,
    )

    expected_pupil_shape = (
        expected_trials,
        120,
        3,
    )

    if eeg_shape != expected_eeg_shape:
        raise RuntimeError(
            f"{expected_subject}: unexpected EEG shape {eeg_shape}."
        )

    if ecg_shape != expected_ecg_shape:
        raise RuntimeError(
            f"{expected_subject}: unexpected ECG shape {ecg_shape}."
        )

    if pupil_shape != expected_pupil_shape:
        raise RuntimeError(
            f"{expected_subject}: unexpected pupil shape {pupil_shape}."
        )

    if not eeg_finite or not ecg_finite:
        raise RuntimeError(
            f"{expected_subject}: EEG or ECG contains nonfinite values."
        )

    if len(metadata) != expected_trials:
        raise RuntimeError(
            f"{expected_subject}: expected {expected_trials} metadata "
            f"rows, observed {len(metadata)}."
        )

    if len(quality) != expected_trials:
        raise RuntimeError(
            f"{expected_subject}: expected {expected_trials} pupil-quality "
            f"rows, observed {len(quality)}."
        )

    class_counts = (
        metadata["span_length"]
        .astype(int)
        .value_counts()
        .sort_index()
        .to_dict()
    )

    if class_counts != {5: 36, 9: 36, 13: 36}:
        raise RuntimeError(
            f"{expected_subject}: unexpected class counts {class_counts}."
        )

    available = (
        metadata["pupil_available"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1"])
        .to_numpy(dtype=bool)
    )

    if not np.isfinite(pupil[available]).all():
        raise RuntimeError(
            f"{expected_subject}: available pupil windows are invalid."
        )

    if (
        (~available).any()
        and not np.isnan(
            pupil[~available]
        ).all()
    ):
        raise RuntimeError(
            f"{expected_subject}: unavailable pupil windows are not NaN."
        )

    trimodal_counts = (
        metadata.loc[
            available,
            "span_length",
        ]
        .astype(int)
        .value_counts()
        .sort_index()
        .to_dict()
    )

    return {
        "subject": expected_subject,
        "memory_trials": int(len(metadata)),
        "eeg_ecg_windows": int(len(metadata)),
        "trimodal_windows": int(available.sum()),
        "pupil_rejected_windows": int(
            (~available).sum()
        ),
        "span5_windows": int(class_counts.get(5, 0)),
        "span9_windows": int(class_counts.get(9, 0)),
        "span13_windows": int(class_counts.get(13, 0)),
        "trimodal_span5": int(
            trimodal_counts.get(5, 0)
        ),
        "trimodal_span9": int(
            trimodal_counts.get(9, 0)
        ),
        "trimodal_span13": int(
            trimodal_counts.get(13, 0)
        ),
        "clock_segment_count": int(
            summary["clock_segment_count"]
        ),
        "median_pupil_valid_fraction": float(
            quality["pupil_valid_fraction"].median()
        ),
        "median_gaze_valid_fraction": float(
            quality["gaze_valid_fraction"].median()
        ),
    }


def run_strict_primary_cohort(
    config: dict[str, Any],
    *,
    output_root: str | Path,
) -> dict[str, Any]:
    """Preprocess and verify all strict-primary participants."""

    manifest_path = resolve_project_path(
        "outputs/revision/manifests/"
        "ds003838/ds003838_subjects.csv"
    )

    manifest = pd.read_csv(
        manifest_path,
        low_memory=False,
    )

    strict_mask = boolean_series(
        manifest["strict_primary_cohort"]
    )

    strict = (
        manifest.loc[strict_mask]
        .sort_values("subject")
        .reset_index(drop=True)
    )

    dataset_config = config["datasets"]["ds003838"]

    expected_subjects = int(
        dataset_config["strict_primary_subjects"]
    )

    expected_per_class = int(
        dataset_config["expected_trials_per_class"]
    )

    expected_trials = expected_per_class * 3
    expected_channels = len(
        config["eeg"]["ds003838_common_channels"]
    )

    if len(strict) != expected_subjects:
        raise RuntimeError(
            f"Expected {expected_subjects} strict-primary participants, "
            f"observed {len(strict)}."
        )

    destination = resolve_project_path(
        output_root
    )

    destination.mkdir(
        parents=True,
        exist_ok=True,
    )

    subject_rows: list[dict[str, Any]] = []
    metadata_frames: list[pd.DataFrame] = []
    quality_frames: list[pd.DataFrame] = []

    warnings.filterwarnings(
        "ignore",
        message=r"Complex objects .* are not supported.*",
        category=UserWarning,
        module=r"pymatreader.*",
    )

    total = len(strict)

    for index, manifest_row in strict.iterrows():
        subject = str(manifest_row["subject"])
        subject_directory = destination / subject

        print(
            f"\n[{index + 1:02d}/{total:02d}] {subject}",
            flush=True,
        )

        if subject_directory.exists():
            print(
                "Existing output found; verifying before skip.",
                flush=True,
            )

        else:
            arrays, metadata, audit = preprocess_subject(
                manifest_row,
                config,
            )

            write_subject_output(
                arrays,
                metadata,
                audit,
                subject_directory,
            )

            print(
                "Preprocessing complete.",
                flush=True,
            )

        verified = verify_subject_output(
            subject_directory,
            expected_subject=subject,
            expected_channels=expected_channels,
            expected_trials=expected_trials,
        )

        subject_rows.append(verified)

        metadata = pd.read_csv(
            subject_directory / "metadata.csv",
            low_memory=False,
        )

        quality = pd.read_csv(
            subject_directory / "pupil_quality.csv",
            low_memory=False,
        )

        metadata_frames.append(metadata)
        quality_frames.append(quality)

        print(
            "Verified: "
            f"EEGECG={verified['eeg_ecg_windows']}, "
            f"trimodal={verified['trimodal_windows']}, "
            f"pupil rejected={verified['pupil_rejected_windows']}",
            flush=True,
        )

    subjects = pd.DataFrame(subject_rows)
    metadata = pd.concat(
        metadata_frames,
        ignore_index=True,
    )

    quality = pd.concat(
        quality_frames,
        ignore_index=True,
    )

    participant_count = int(
        subjects["subject"].nunique()
    )

    eeg_ecg_windows = int(
        subjects["eeg_ecg_windows"].sum()
    )

    trimodal_windows = int(
        subjects["trimodal_windows"].sum()
    )

    class_counts = Counter(
        metadata["span_length"].astype(int)
    )

    available = (
        metadata["pupil_available"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1"])
    )

    trimodal_class_counts = Counter(
        metadata.loc[
            available,
            "span_length",
        ].astype(int)
    )

    expected_total = (
        expected_subjects
        * expected_trials
    )

    expected_class_total = (
        expected_subjects
        * expected_per_class
    )

    if participant_count != expected_subjects:
        raise RuntimeError(
            "Final strict-primary participant count changed."
        )

    if eeg_ecg_windows != expected_total:
        raise RuntimeError(
            f"Expected {expected_total} EEGECG windows, "
            f"observed {eeg_ecg_windows}."
        )

    if class_counts != Counter(
        {
            5: expected_class_total,
            9: expected_class_total,
            13: expected_class_total,
        }
    ):
        raise RuntimeError(
            f"Strict-primary classes are not balanced: "
            f"{dict(class_counts)}"
        )

    if metadata["segment_id"].duplicated().any():
        raise RuntimeError(
            "Duplicate segment identifiers detected."
        )

    if set(metadata["participant"]) != set(
        subjects["subject"]
    ):
        raise RuntimeError(
            "Metadata participants differ from the subject registry."
        )

    summary = {
        "dataset": "ds003838",
        "cohort": "strict_primary",
        "participant_count": participant_count,
        "expected_trials_per_class_per_participant":
            expected_per_class,
        "eeg_ecg_window_count": eeg_ecg_windows,
        "trimodal_window_count": trimodal_windows,
        "pupil_rejected_window_count": int(
            eeg_ecg_windows - trimodal_windows
        ),
        "class_window_counts": {
            str(key): int(value)
            for key, value in sorted(
                class_counts.items()
            )
        },
        "trimodal_class_window_counts": {
            str(key): int(value)
            for key, value in sorted(
                trimodal_class_counts.items()
            )
        },
        "participants_with_all_108_trimodal_windows": int(
            (
                subjects["trimodal_windows"]
                == expected_trials
            ).sum()
        ),
        "participants_with_at_least_one_pupil_rejection": int(
            (
                subjects["pupil_rejected_windows"]
                > 0
            ).sum()
        ),
        "minimum_trimodal_windows_per_participant": int(
            subjects["trimodal_windows"].min()
        ),
        "median_trimodal_windows_per_participant": float(
            subjects["trimodal_windows"].median()
        ),
        "maximum_trimodal_windows_per_participant": int(
            subjects["trimodal_windows"].max()
        ),
        "median_pupil_valid_fraction_across_windows": float(
            quality["pupil_valid_fraction"].median()
        ),
        "median_gaze_valid_fraction_across_windows": float(
            quality["gaze_valid_fraction"].median()
        ),
        "window_definition": {
            "anchor": "final_digit_offset",
            "start_offset_seconds": float(
                dataset_config[
                    "window"
                ][
                    "start_offset_seconds"
                ]
            ),
            "duration_seconds": float(
                dataset_config[
                    "window"
                ][
                    "duration_seconds"
                ]
            ),
        },
        "sampling_rates_hz": {
            "eeg": 128.0,
            "ecg": 128.0,
            "pupil": 30.0,
        },
        "eeg_channels": list(
            config["eeg"]["ds003838_common_channels"]
        ),
        "ecg_channel": "ECG",
        "pupil_missingness_policy": (
            "All EEGECG windows retained; unavailable pupil "
            "windows represented by NaN and excluded only from "
            "pupil-dependent modality combinations."
        ),
    }

    temporary_summary = destination / (
        "cohort_summary.json.building"
    )

    temporary_subjects = destination / (
        "cohort_subjects.csv.building"
    )

    temporary_metadata = destination / (
        "metadata.csv.building"
    )

    temporary_quality = destination / (
        "pupil_quality.csv.building"
    )

    temporary_summary.write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )

    subjects.to_csv(
        temporary_subjects,
        index=False,
    )

    metadata.to_csv(
        temporary_metadata,
        index=False,
    )

    quality.to_csv(
        temporary_quality,
        index=False,
    )

    temporary_summary.replace(
        destination / "cohort_summary.json"
    )

    temporary_subjects.replace(
        destination / "cohort_subjects.csv"
    )

    temporary_metadata.replace(
        destination / "metadata.csv"
    )

    temporary_quality.replace(
        destination / "pupil_quality.csv"
    )

    print("\n===== STRICT-PRIMARY COHORT COMPLETE =====")
    print(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        )
    )

    print("\nOutput directory:", destination)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run resumable strict-primary ds003838 preprocessing."
        )
    )

    parser.add_argument(
        "--output",
        default=(
            "outputs/revision/preprocessed/"
            "ds003838_strict_primary"
        ),
    )

    arguments = parser.parse_args()

    config = load_revision_config(
        check_input_paths=True
    )

    run_strict_primary_cohort(
        config,
        output_root=arguments.output,
    )


if __name__ == "__main__":
    main()