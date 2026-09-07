"""Read-only manifest construction for OpenNeuro ds003838."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from src.core.config import load_revision_config


def _natural_identifier_key(value: str) -> tuple[str, int]:
    match = re.search(r"(\d+)$", value)
    number = int(match.group(1)) if match else -1
    prefix = value[: match.start(1)] if match else value
    return prefix.casefold(), number


def _is_yes(value: object) -> bool:
    return str(value).strip().casefold() == "yes"


def _decode_code(code: int) -> dict[str, Any]:
    text = str(int(code))

    if len(text) == 6 and text.startswith("5"):
        return {
            "condition": "control",
            "position": int(text[1:4]),
            "length": int(text[4:6]),
            "correct": None,
        }

    if len(text) == 7 and text.startswith("6"):
        return {
            "condition": "memory",
            "position": int(text[1:4]),
            "length": int(text[4:6]),
            "correct": int(text[6]),
        }

    raise ValueError(f"Unsupported ds003838 event code: {code}")


def _read_channel_names(path: Path) -> list[str]:
    frame = pd.read_csv(path, sep="\t")

    if "name" not in frame.columns:
        return []

    return [
        str(value).strip()
        for value in frame["name"].tolist()
        if str(value).strip()
    ]


def _prepare_events(
    path: Path,
    *,
    code_column: str,
    time_column: str,
) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t")

    if code_column not in frame.columns:
        raise ValueError(
            f"Missing event-code column '{code_column}' in {path}"
        )

    if time_column not in frame.columns:
        raise ValueError(
            f"Missing event-time column '{time_column}' in {path}"
        )

    frame["code"] = pd.to_numeric(
        frame[code_column],
        errors="coerce",
    )

    frame = (
        frame.loc[frame["code"].notna()]
        .copy()
        .reset_index(drop=True)
    )

    frame["code"] = frame["code"].astype(np.int64)
    frame["time"] = pd.to_numeric(
        frame[time_column],
        errors="raise",
    ).astype(float)

    decoded = frame["code"].map(_decode_code)

    frame["condition"] = decoded.map(
        lambda item: item["condition"]
    )
    frame["position"] = decoded.map(
        lambda item: item["position"]
    )
    frame["length"] = decoded.map(
        lambda item: item["length"]
    )

    return frame


def _build_trials(events: pd.DataFrame) -> list[dict[str, Any]]:
    trials: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []

    for row in events.to_dict("records"):
        position = int(row["position"])
        length = int(row["length"])

        if position == 1:
            if current:
                raise RuntimeError(
                    "A new trial started before the previous trial ended."
                )

            current = [row]

        else:
            if not current:
                raise RuntimeError(
                    "A trial item appeared without a trial start."
                )

            previous = current[-1]

            if (
                row["condition"] != previous["condition"]
                or length != int(previous["length"])
                or position != int(previous["position"]) + 1
            ):
                raise RuntimeError(
                    "A non-contiguous trial event sequence was detected."
                )

            current.append(row)

        if position == length:
            if len(current) != length:
                raise RuntimeError(
                    "Trial length does not match its event count."
                )

            trials.append(
                {
                    "condition": row["condition"],
                    "length": length,
                    "events": current,
                    "signature": (
                        row["condition"],
                        length,
                        tuple(
                            int(event["code"])
                            for event in current
                        ),
                    ),
                    "start_time": float(current[0]["time"]),
                    "final_time": float(current[-1]["time"]),
                    "final_duration": float(
                        current[-1].get("duration", 0.0)
                    ),
                    "final_offset": (
                        float(current[-1]["time"])
                        + float(
                            current[-1].get("duration", 0.0)
                        )
                    ),
                }
            )

            current = []

    if current:
        raise RuntimeError(
            "An incomplete trial remained at the end of the event file."
        )

    return trials


def _align_trials(
    eeg_trials: list[dict[str, Any]],
    pupil_trials: list[dict[str, Any]],
) -> list[tuple[int, int]]:
    """Find the ordered pupil-trial subsequence in EEG events."""

    eeg_count = len(eeg_trials)
    pupil_count = len(pupil_trials)

    dynamic = np.zeros(
        (eeg_count + 1, pupil_count + 1),
        dtype=np.int16,
    )

    for eeg_index in range(eeg_count - 1, -1, -1):
        for pupil_index in range(pupil_count - 1, -1, -1):
            if (
                eeg_trials[eeg_index]["signature"]
                == pupil_trials[pupil_index]["signature"]
            ):
                dynamic[eeg_index, pupil_index] = (
                    1
                    + dynamic[
                        eeg_index + 1,
                        pupil_index + 1,
                    ]
                )
            else:
                dynamic[eeg_index, pupil_index] = max(
                    dynamic[eeg_index + 1, pupil_index],
                    dynamic[eeg_index, pupil_index + 1],
                )

    alignment: list[tuple[int, int]] = []
    eeg_index = 0
    pupil_index = 0

    while eeg_index < eeg_count and pupil_index < pupil_count:
        if (
            eeg_trials[eeg_index]["signature"]
            == pupil_trials[pupil_index]["signature"]
        ):
            alignment.append((eeg_index, pupil_index))
            eeg_index += 1
            pupil_index += 1

        elif (
            dynamic[eeg_index + 1, pupil_index]
            >= dynamic[eeg_index, pupil_index + 1]
        ):
            eeg_index += 1

        else:
            pupil_index += 1

    return alignment


def _affine_fit(
    eeg_times: np.ndarray,
    pupil_times: np.ndarray,
) -> dict[str, Any]:
    if len(eeg_times) < 2:
        raise ValueError(
            "At least two aligned events are required for clock fitting."
        )

    eeg_mean = float(eeg_times.mean())
    pupil_mean = float(pupil_times.mean())

    denominator = float(
        np.sum((eeg_times - eeg_mean) ** 2)
    )

    if denominator <= 0:
        raise ValueError("Clock-fit EEG times have zero variance.")

    slope = float(
        np.sum(
            (eeg_times - eeg_mean)
            * (pupil_times - pupil_mean)
        )
        / denominator
    )

    intercept = pupil_mean - slope * eeg_mean
    residuals = pupil_times - (
        intercept + slope * eeg_times
    )

    return {
        "slope": slope,
        "intercept": intercept,
        "residuals": residuals,
        "maximum_absolute_residual": float(
            np.max(np.abs(residuals))
        ),
        "median_absolute_residual": float(
            np.median(np.abs(residuals))
        ),
        "rmse": float(
            np.sqrt(np.mean(residuals ** 2))
        ),
    }


def _clock_segments(
    eeg_times: np.ndarray,
    pupil_times: np.ndarray,
    *,
    jump_threshold_seconds: float,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    offsets = pupil_times - eeg_times
    offset_jumps = np.diff(offsets)

    jump_indices = np.flatnonzero(
        np.abs(offset_jumps) > jump_threshold_seconds
    )

    split_points = [0]
    split_points.extend(
        int(index + 1)
        for index in jump_indices
    )
    split_points.append(len(eeg_times))

    segments: list[dict[str, Any]] = []

    for start, stop in zip(
        split_points[:-1],
        split_points[1:],
    ):
        if stop - start < 3:
            raise RuntimeError(
                "A clock segment contains fewer than three events."
            )

        fit = _affine_fit(
            eeg_times[start:stop],
            pupil_times[start:stop],
        )

        segments.append(
            {
                "start_index": start,
                "stop_index": stop,
                "eeg_start": float(eeg_times[start]),
                "eeg_stop": float(eeg_times[stop - 1]),
                "slope": fit["slope"],
                "intercept": fit["intercept"],
                "maximum_absolute_residual":
                    fit["maximum_absolute_residual"],
                "median_absolute_residual":
                    fit["median_absolute_residual"],
                "rmse": fit["rmse"],
            }
        )

    return segments, offset_jumps


def _map_eeg_time(
    eeg_time: float,
    segments: list[dict[str, Any]],
) -> float:
    if not segments:
        raise ValueError("No synchronization segments are available.")

    if len(segments) == 1:
        segment = segments[0]

    else:
        segment = segments[-1]

        for current, following in zip(
            segments[:-1],
            segments[1:],
        ):
            boundary = (
                float(current["eeg_stop"])
                + float(following["eeg_start"])
            ) / 2.0

            if eeg_time <= boundary:
                segment = current
                break

    return (
        float(segment["intercept"])
        + float(segment["slope"]) * eeg_time
    )


def _timestamp_coverage_segments(
    path: Path,
    *,
    maximum_gap_seconds: float,
) -> tuple[list[tuple[float, float]], float]:
    """Identify continuous pupil-sample intervals from its timestamp column."""

    if maximum_gap_seconds <= 0:
        raise ValueError(
            "maximum_gap_seconds must be greater than zero."
        )

    segments: list[tuple[float, float]] = []

    segment_start: float | None = None
    previous_timestamp: float | None = None
    maximum_observed_gap = 0.0

    for chunk in pd.read_csv(
        path,
        sep="	",
        usecols=["pupil_timestamp"],
        chunksize=500_000,
        low_memory=False,
    ):
        timestamps = pd.to_numeric(
            chunk["pupil_timestamp"],
            errors="coerce",
        ).to_numpy(dtype=float)

        timestamps = timestamps[np.isfinite(timestamps)]

        for raw_timestamp in timestamps:
            timestamp = float(raw_timestamp)

            if previous_timestamp is None:
                segment_start = timestamp
                previous_timestamp = timestamp
                continue

            # Duplicate or slightly out-of-order rows may occur because
            # multiple eye/detection records share nearly identical times.
            if timestamp < previous_timestamp:
                continue

            gap = timestamp - previous_timestamp
            maximum_observed_gap = max(
                maximum_observed_gap,
                gap,
            )

            if gap > maximum_gap_seconds:
                if segment_start is None:
                    raise RuntimeError(
                        "Missing pupil coverage-segment start."
                    )

                segments.append(
                    (
                        float(segment_start),
                        float(previous_timestamp),
                    )
                )

                segment_start = timestamp

            previous_timestamp = timestamp

    if (
        segment_start is None
        or previous_timestamp is None
    ):
        raise RuntimeError(
            f"No numeric pupil timestamps found in {path}"
        )

    segments.append(
        (
            float(segment_start),
            float(previous_timestamp),
        )
    )

    return segments, maximum_observed_gap


def _window_inside_coverage(
    window_start: float,
    window_stop: float,
    coverage_segments: list[tuple[float, float]],
) -> bool:
    """Return whether a window lies inside one continuous sample interval."""

    return any(
        window_start >= segment_start
        and window_stop <= segment_stop
        for segment_start, segment_stop in coverage_segments
    )


def scan_ds003838(
    dataset_root: str | Path,
    *,
    required_channels: Iterable[str],
    expected_trials_per_class: int = 36,
    window_start_offset_seconds: float = 1.0,
    window_duration_seconds: float = 4.0,
    maximum_global_residual_seconds: float = 0.100,
    maximum_slope_difference: float = 0.001,
    clock_jump_threshold_seconds: float = 0.250,
    maximum_pupil_sample_gap_seconds: float = 0.500,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Scan ds003838 without modifying or loading signal arrays."""

    root = Path(dataset_root).expanduser().resolve(strict=True)
    participants_path = root / "participants.tsv"

    participants = pd.read_csv(
        participants_path,
        sep="\t",
    )

    required = [
        str(channel).strip()
        for channel in required_channels
    ]
    required_folded = {
        channel.casefold(): channel
        for channel in required
    }

    records: list[dict[str, Any]] = []

    for _, participant in participants.iterrows():
        subject = str(
            participant["participant_id"]
        ).strip()

        subject_root = root / subject

        paths = {
            "eeg": (
                subject_root / "eeg" /
                f"{subject}_task-memory_eeg.set"
            ),
            "eeg_channels": (
                subject_root / "eeg" /
                f"{subject}_task-memory_channels.tsv"
            ),
            "eeg_events": (
                subject_root / "eeg" /
                f"{subject}_task-memory_events.tsv"
            ),
            "ecg": (
                subject_root / "ecg" /
                f"{subject}_task-memory_ecg.set"
            ),
            "pupil": (
                subject_root / "pupil" /
                f"{subject}_task-memory_pupil.tsv"
            ),
            "pupil_events": (
                subject_root / "pupil" /
                f"{subject}_task-memory_events.tsv"
            ),
            "behavior": (
                subject_root / "beh" /
                f"{subject}_task-memory_beh.tsv"
            ),
        }

        excluded_modalities = [
            name
            for name, column in (
                ("EEG", "EEG_excluded"),
                ("ECG", "ECG_excluded"),
                ("Pupil", "pupil_excluded"),
                ("Behavior", "behavior_excluded"),
            )
            if _is_yes(participant.get(column, ""))
        ]

        missing_files = [
            name
            for name, path in paths.items()
            if not path.is_file()
        ]

        available_channels: list[str] = []

        if paths["eeg_channels"].is_file():
            available_channels = _read_channel_names(
                paths["eeg_channels"]
            )

        available_folded = {
            channel.casefold()
            for channel in available_channels
        }

        missing_channels = [
            original
            for folded, original in required_folded.items()
            if folded not in available_folded
        ]

        complete_multimodal_files = not missing_files
        required_channels_available = (
            bool(available_channels)
            and not missing_channels
        )

        analysis_eligible = (
            complete_multimodal_files
            and required_channels_available
            and not excluded_modalities
        )

        record: dict[str, Any] = {
            "dataset": "ds003838",
            "subject": subject,
            "complete_multimodal_files":
                complete_multimodal_files,
            "required_channels_available":
                required_channels_available,
            "analysis_eligible": analysis_eligible,
            "strict_primary_cohort": False,
            "sensitivity_cohort": False,
            "piecewise_sync_required": False,
            "excluded_modalities":
                ";".join(excluded_modalities),
            "missing_files": ";".join(missing_files),
            "available_eeg_channel_count":
                len(available_channels),
            "missing_required_channels":
                ";".join(missing_channels),
            "eeg_trial_count": 0,
            "pupil_trial_count": 0,
            "matched_trial_count": 0,
            "usable_memory_5": 0,
            "usable_memory_9": 0,
            "usable_memory_13": 0,
            "usable_memory_total": 0,
            "complete_balanced_108": False,
            "clock_segment_count": 0,
            "clock_jump_count": 0,
            "global_clock_slope": None,
            "global_maximum_residual_ms": None,
            "global_median_residual_ms": None,
            "piecewise_maximum_residual_ms": None,
            "global_sync_pass": False,
            "pupil_span_seconds": None,
            "pupil_coverage_segment_count": 0,
            "maximum_pupil_sample_gap_seconds": None,
            "eeg_path": str(paths["eeg"]),
            "ecg_path": str(paths["ecg"]),
            "pupil_path": str(paths["pupil"]),
            "eeg_events_path": str(paths["eeg_events"]),
            "pupil_events_path": str(paths["pupil_events"]),
            "behavior_path": str(paths["behavior"]),
        }

        if analysis_eligible:
            eeg_events = _prepare_events(
                paths["eeg_events"],
                code_column="value",
                time_column="onset",
            )

            pupil_events = _prepare_events(
                paths["pupil_events"],
                code_column="label",
                time_column="timestamp",
            )

            eeg_trials = _build_trials(eeg_events)
            pupil_trials = _build_trials(pupil_events)

            alignment = _align_trials(
                eeg_trials,
                pupil_trials,
            )

            if len(alignment) != len(pupil_trials):
                raise RuntimeError(
                    f"{subject}: pupil trials are not an ordered "
                    "subsequence of EEG trials."
                )

            matched_eeg_indices = {
                eeg_index
                for eeg_index, _ in alignment
            }

            eeg_pair_times: list[float] = []
            pupil_pair_times: list[float] = []

            for eeg_index, pupil_index in alignment:
                eeg_trial = eeg_trials[eeg_index]
                pupil_trial = pupil_trials[pupil_index]

                for eeg_event, pupil_event in zip(
                    eeg_trial["events"],
                    pupil_trial["events"],
                ):
                    if (
                        int(eeg_event["code"])
                        != int(pupil_event["code"])
                    ):
                        raise RuntimeError(
                            f"{subject}: aligned event codes differ."
                        )

                    eeg_pair_times.append(
                        float(eeg_event["time"])
                    )
                    pupil_pair_times.append(
                        float(pupil_event["time"])
                    )

            eeg_times = np.asarray(
                eeg_pair_times,
                dtype=float,
            )
            pupil_times = np.asarray(
                pupil_pair_times,
                dtype=float,
            )

            global_fit = _affine_fit(
                eeg_times,
                pupil_times,
            )

            segments, offset_jumps = _clock_segments(
                eeg_times,
                pupil_times,
                jump_threshold_seconds=(
                    clock_jump_threshold_seconds
                ),
            )

            (
                pupil_coverage_segments,
                maximum_observed_pupil_gap,
            ) = _timestamp_coverage_segments(
                paths["pupil"],
                maximum_gap_seconds=(
                    maximum_pupil_sample_gap_seconds
                ),
            )

            pupil_first = pupil_coverage_segments[0][0]
            pupil_last = pupil_coverage_segments[-1][1]

            usable_by_length = {
                5: 0,
                9: 0,
                13: 0,
            }

            for eeg_index, trial in enumerate(eeg_trials):
                if trial["condition"] != "memory":
                    continue

                eeg_window_start = (
                    float(trial["final_offset"])
                    + window_start_offset_seconds
                )
                eeg_window_stop = (
                    eeg_window_start
                    + window_duration_seconds
                )

                pupil_window_start = _map_eeg_time(
                    eeg_window_start,
                    segments,
                )
                pupil_window_stop = _map_eeg_time(
                    eeg_window_stop,
                    segments,
                )

                if _window_inside_coverage(
                    pupil_window_start,
                    pupil_window_stop,
                    pupil_coverage_segments,
                ):
                    length = int(trial["length"])

                    if length in usable_by_length:
                        usable_by_length[length] += 1

            usable_total = sum(usable_by_length.values())

            complete_balanced = all(
                usable_by_length[length]
                == expected_trials_per_class
                for length in (5, 9, 13)
            )

            clock_jump_count = int(
                (
                    np.abs(offset_jumps)
                    > clock_jump_threshold_seconds
                ).sum()
            )

            global_sync_pass = (
                clock_jump_count == 0
                and abs(global_fit["slope"] - 1.0)
                <= maximum_slope_difference
                and global_fit["maximum_absolute_residual"]
                <= maximum_global_residual_seconds
            )

            piecewise_maximum_residual = max(
                float(segment["maximum_absolute_residual"])
                for segment in segments
            )

            strict_primary = (
                complete_balanced
                and global_sync_pass
            )

            record.update(
                {
                    "strict_primary_cohort": strict_primary,
                    "sensitivity_cohort": usable_total > 0,
                    "piecewise_sync_required":
                        clock_jump_count > 0,
                    "eeg_trial_count": len(eeg_trials),
                    "pupil_trial_count": len(pupil_trials),
                    "matched_trial_count": len(alignment),
                    "usable_memory_5":
                        usable_by_length[5],
                    "usable_memory_9":
                        usable_by_length[9],
                    "usable_memory_13":
                        usable_by_length[13],
                    "usable_memory_total": usable_total,
                    "complete_balanced_108":
                        complete_balanced,
                    "clock_segment_count": len(segments),
                    "clock_jump_count": clock_jump_count,
                    "global_clock_slope":
                        global_fit["slope"],
                    "global_maximum_residual_ms": (
                        1000
                        * global_fit[
                            "maximum_absolute_residual"
                        ]
                    ),
                    "global_median_residual_ms": (
                        1000
                        * global_fit[
                            "median_absolute_residual"
                        ]
                    ),
                    "piecewise_maximum_residual_ms": (
                        1000 * piecewise_maximum_residual
                    ),
                    "global_sync_pass": global_sync_pass,
                    "pupil_span_seconds":
                        pupil_last - pupil_first,
                    "pupil_coverage_segment_count":
                        len(pupil_coverage_segments),
                    "maximum_pupil_sample_gap_seconds":
                        maximum_observed_pupil_gap,
                }
            )

        records.append(record)

    records.sort(
        key=lambda record: _natural_identifier_key(
            str(record["subject"])
        )
    )

    eligible_records = [
        record
        for record in records
        if record["analysis_eligible"]
    ]

    strict_records = [
        record
        for record in records
        if record["strict_primary_cohort"]
    ]

    sensitivity_records = [
        record
        for record in records
        if record["sensitivity_cohort"]
    ]

    piecewise_records = [
        record
        for record in records
        if record["piecewise_sync_required"]
    ]

    incomplete_records = [
        record
        for record in eligible_records
        if not record["complete_balanced_108"]
    ]

    sensitivity_class_totals = {
        str(length): sum(
            int(record[f"usable_memory_{length}"])
            for record in sensitivity_records
        )
        for length in (5, 9, 13)
    }

    summary = {
        "dataset": "ds003838",
        "dataset_root": str(root),
        "required_eeg_channels": required,
        "expected_trials_per_class":
            expected_trials_per_class,
        "window_definition": {
            "condition": "memory",
            "anchor": "final_digit_offset",
            "start_offset_seconds":
                window_start_offset_seconds,
            "duration_seconds":
                window_duration_seconds,
        },
        "synchronization_thresholds": {
            "maximum_global_residual_seconds":
                maximum_global_residual_seconds,
            "maximum_slope_difference":
                maximum_slope_difference,
            "clock_jump_threshold_seconds":
                clock_jump_threshold_seconds,
            "maximum_pupil_sample_gap_seconds":
                maximum_pupil_sample_gap_seconds,
        },
        "participant_rows": len(records),
        "analysis_eligible_subject_count":
            len(eligible_records),
        "strict_primary_subject_count":
            len(strict_records),
        "strict_primary_window_count": sum(
            int(record["usable_memory_total"])
            for record in strict_records
        ),
        "strict_primary_subjects": [
            str(record["subject"])
            for record in strict_records
        ],
        "sensitivity_subject_count":
            len(sensitivity_records),
        "sensitivity_window_count": sum(
            int(record["usable_memory_total"])
            for record in sensitivity_records
        ),
        "sensitivity_class_window_counts":
            sensitivity_class_totals,
        "sensitivity_subjects": [
            str(record["subject"])
            for record in sensitivity_records
        ],
        "piecewise_sync_subject_count":
            len(piecewise_records),
        "piecewise_sync_subjects": [
            str(record["subject"])
            for record in piecewise_records
        ],
        "incomplete_pupil_subject_count":
            len(incomplete_records),
        "incomplete_pupil_subjects": [
            str(record["subject"])
            for record in incomplete_records
        ],
    }

    return records, summary


def write_ds003838_manifest(
    records: list[dict[str, Any]],
    summary: Mapping[str, Any],
    output_directory: str | Path,
) -> tuple[Path, Path]:
    """Write a subject-level CSV manifest and JSON summary."""

    output_directory = Path(
        output_directory
    ).resolve(strict=False)

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_path = (
        output_directory
        / "ds003838_subjects.csv"
    )

    json_path = (
        output_directory
        / "ds003838_summary.json"
    )

    fieldnames = [
        "dataset",
        "subject",
        "complete_multimodal_files",
        "required_channels_available",
        "analysis_eligible",
        "strict_primary_cohort",
        "sensitivity_cohort",
        "piecewise_sync_required",
        "excluded_modalities",
        "missing_files",
        "available_eeg_channel_count",
        "missing_required_channels",
        "eeg_trial_count",
        "pupil_trial_count",
        "matched_trial_count",
        "usable_memory_5",
        "usable_memory_9",
        "usable_memory_13",
        "usable_memory_total",
        "complete_balanced_108",
        "clock_segment_count",
        "clock_jump_count",
        "global_clock_slope",
        "global_maximum_residual_ms",
        "global_median_residual_ms",
        "piecewise_maximum_residual_ms",
        "global_sync_pass",
        "pupil_span_seconds",
        "pupil_coverage_segment_count",
        "maximum_pupil_sample_gap_seconds",
        "eeg_path",
        "ecg_path",
        "pupil_path",
        "eeg_events_path",
        "pupil_events_path",
        "behavior_path",
    ]

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(records)

    json_path.write_text(
        json.dumps(
            dict(summary),
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    return csv_path, json_path


def _main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build the read-only ds003838 subject manifest."
        )
    )

    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Audit the dataset without writing manifest files.",
    )

    args = parser.parse_args()

    config = load_revision_config(
        check_input_paths=True
    )

    dataset_config = config["datasets"]["ds003838"]
    synchronization = dataset_config["synchronization"]
    window = dataset_config["window"]

    records, summary = scan_ds003838(
        config["paths"]["ds003838_root"],
        required_channels=(
            config["eeg"]["ds003838_common_channels"]
        ),
        expected_trials_per_class=int(
            dataset_config["expected_trials_per_class"]
        ),
        window_start_offset_seconds=float(
            window["start_offset_seconds"]
        ),
        window_duration_seconds=float(
            window["duration_seconds"]
        ),
        maximum_global_residual_seconds=float(
            synchronization[
                "maximum_global_residual_seconds"
            ]
        ),
        maximum_slope_difference=float(
            synchronization[
                "maximum_slope_difference"
            ]
        ),
        clock_jump_threshold_seconds=float(
            synchronization[
                "clock_jump_threshold_seconds"
            ]
        ),
        maximum_pupil_sample_gap_seconds=float(
            synchronization[
                "maximum_pupil_sample_gap_seconds"
            ]
        ),
    )

    print("===== ds003838 MANIFEST SUMMARY =====")
    print(
        "Participant rows:",
        summary["participant_rows"],
    )
    print(
        "Analysis-eligible subjects:",
        summary["analysis_eligible_subject_count"],
    )
    print(
        "Strict primary subjects:",
        summary["strict_primary_subject_count"],
    )
    print(
        "Strict primary windows:",
        summary["strict_primary_window_count"],
    )
    print(
        "Sensitivity subjects:",
        summary["sensitivity_subject_count"],
    )
    print(
        "Sensitivity windows:",
        summary["sensitivity_window_count"],
    )
    print(
        "Sensitivity class counts:",
        summary["sensitivity_class_window_counts"],
    )
    print(
        "Piecewise-sync subjects:",
        ", ".join(
            summary["piecewise_sync_subjects"]
        )
        or "[NONE]",
    )
    print(
        "Incomplete-pupil subjects:",
        ", ".join(
            summary["incomplete_pupil_subjects"]
        )
        or "[NONE]",
    )
    print(
        "Strict primary subject list:",
        ", ".join(
            summary["strict_primary_subjects"]
        ),
    )

    if not args.no_write:
        manifest_directory = (
            Path(config["paths"]["artifacts"])
            / "manifests"
        )

        csv_path, json_path = write_ds003838_manifest(
            records,
            summary,
            manifest_directory,
        )

        print("CSV manifest:", csv_path)
        print("JSON summary:", json_path)


if __name__ == "__main__":
    _main()
