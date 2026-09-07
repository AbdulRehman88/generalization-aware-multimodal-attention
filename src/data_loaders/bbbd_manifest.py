"""Read-only manifest construction for BBBD Experiment 2 and Experiment 3."""

from __future__ import annotations

import argparse
import csv
import json
import re
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

from src.core.config import load_revision_config


_PATH_PATTERNS = {
    "eeg": re.compile(
        r"^(sub-\d+)/(ses-\d+)/eeg/"
        r".+_task-(stim\d+)_eeg\.bdf$",
        re.IGNORECASE,
    ),
    "channels": re.compile(
        r"^(sub-\d+)/(ses-\d+)/eeg/"
        r".+_task-(stim\d+)_channels\.tsv$",
        re.IGNORECASE,
    ),
    "ecg": re.compile(
        r"^(sub-\d+)/(ses-\d+)/beh/"
        r".+_task-(stim\d+)_recording-ecg_physio\.tsv\.gz$",
        re.IGNORECASE,
    ),
    "pupil": re.compile(
        r"^(sub-\d+)/(ses-\d+)/eyetrack/"
        r".+_task-(stim\d+)_pupil_eyetrack\.tsv\.gz$",
        re.IGNORECASE,
    ),
    "events": re.compile(
        r"^(sub-\d+)/(ses-\d+)/eeg/"
        r".+_task-(stim\d+)_events\.tsv$",
        re.IGNORECASE,
    ),
}


def _natural_identifier_key(value: str) -> tuple[str, int]:
    match = re.search(r"(\d+)$", value)
    number = int(match.group(1)) if match else -1
    prefix = value[: match.start(1)] if match else value
    return prefix.casefold(), number


def _expected_task_ids(expected_tasks: int) -> list[str]:
    if expected_tasks < 1:
        raise ValueError("expected_tasks must be at least 1.")
    return [f"stim{index:02d}" for index in range(1, expected_tasks + 1)]


def _match_raw_path(
    path: str,
) -> tuple[str, str, str, str] | None:
    normalized = path.replace("\\", "/")

    if normalized.startswith("derivatives/"):
        return None

    for modality, pattern in _PATH_PATTERNS.items():
        match = pattern.match(normalized)
        if match:
            subject, session, task = match.groups()
            return subject, session, task, modality

    return None


def _read_channel_names(
    archive: zipfile.ZipFile,
    member_name: str,
) -> list[str]:
    with archive.open(member_name, "r") as handle:
        text = handle.read().decode("utf-8-sig", errors="replace")

    lines = [line.strip() for line in text.splitlines() if line.strip()]

    if len(lines) < 2:
        return []

    channels: list[str] = []

    for line in lines[1:]:
        if "\t" in line:
            name = line.split("\t", 1)[0].strip()
        else:
            fields = line.split()
            name = fields[0].strip() if fields else ""

        if name:
            channels.append(name)

    return channels


def _distribution(
    records: Iterable[Mapping[str, Any]],
) -> dict[str, int]:
    counter = Counter(
        f"{record['session']}|{record['task']}"
        for record in records
        if record["complete_multimodal"]
    )
    return dict(sorted(counter.items()))


def scan_bbbd_archive(
    archive_path: str | Path,
    *,
    dataset_name: str,
    expected_tasks: int,
    required_channels: Iterable[str],
    label_map: Mapping[str, int],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Scan a BBBD archive without extracting signal files."""

    archive_path = Path(archive_path).expanduser().resolve(strict=True)

    if not zipfile.is_zipfile(archive_path):
        raise zipfile.BadZipFile(f"Not a valid ZIP archive: {archive_path}")

    required = [str(channel).strip() for channel in required_channels]
    required_casefold = {channel.casefold(): channel for channel in required}
    expected_task_ids = _expected_task_ids(expected_tasks)

    records_by_key: dict[
        tuple[str, str, str],
        dict[str, Any],
    ] = {}

    with zipfile.ZipFile(archive_path, "r", allowZip64=True) as archive:
        for member in archive.infolist():
            match = _match_raw_path(member.filename)

            if match is None:
                continue

            subject, session, task, modality = match
            key = (subject, session, task)

            record = records_by_key.setdefault(
                key,
                {
                    "dataset": dataset_name,
                    "subject": subject,
                    "session": session,
                    "task": task,
                    "label": label_map.get(session),
                    "eeg_path": "",
                    "channels_path": "",
                    "ecg_path": "",
                    "pupil_path": "",
                    "events_path": "",
                },
            )

            path_key = f"{modality}_path"
            record[path_key] = member.filename.replace("\\", "/")

        for record in records_by_key.values():
            record["complete_multimodal"] = bool(
                record["eeg_path"]
                and record["ecg_path"]
                and record["pupil_path"]
            )

            available_channels: list[str] = []

            if record["channels_path"]:
                available_channels = _read_channel_names(
                    archive,
                    record["channels_path"],
                )

            available_casefold = {
                channel.casefold() for channel in available_channels
            }

            missing_channels = [
                original
                for folded, original in required_casefold.items()
                if folded not in available_casefold
            ]

            record["available_eeg_channel_count"] = len(available_channels)
            record["missing_required_channels"] = ";".join(missing_channels)
            record["required_channels_available"] = (
                bool(record["channels_path"]) and not missing_channels
            )
            record["analysis_eligible"] = (
                record["complete_multimodal"]
                and record["required_channels_available"]
                and record["label"] is not None
            )

    records = sorted(
        records_by_key.values(),
        key=lambda record: (
            _natural_identifier_key(record["subject"]),
            _natural_identifier_key(record["session"]),
            _natural_identifier_key(record["task"]),
        ),
    )

    eligible_keys_by_subject: dict[
        str,
        set[tuple[str, str]],
    ] = {}

    for record in records:
        if record["analysis_eligible"]:
            eligible_keys_by_subject.setdefault(
                record["subject"],
                set(),
            ).add((record["session"], record["task"]))

    required_record_keys = {
        (session, task)
        for session in ("ses-01", "ses-02")
        for task in expected_task_ids
    }

    primary_subjects = sorted(
        [
            subject
            for subject, available_keys in eligible_keys_by_subject.items()
            if required_record_keys.issubset(available_keys)
        ],
        key=_natural_identifier_key,
    )
    primary_subject_set = set(primary_subjects)

    for record in records:
        record["primary_cohort"] = (
            record["subject"] in primary_subject_set
            and record["session"] in {"ses-01", "ses-02"}
            and record["task"] in expected_task_ids
            and record["analysis_eligible"]
        )

    complete_records = [
        record for record in records if record["complete_multimodal"]
    ]
    eligible_records = [
        record for record in records if record["analysis_eligible"]
    ]
    primary_records = [
        record for record in records if record["primary_cohort"]
    ]

    subjects_with_complete = sorted(
        {record["subject"] for record in complete_records},
        key=_natural_identifier_key,
    )

    incompatible_complete = [
        record
        for record in complete_records
        if not record["required_channels_available"]
    ]

    summary = {
        "dataset": dataset_name,
        "archive": str(archive_path),
        "archive_size_bytes": archive_path.stat().st_size,
        "expected_tasks_per_session": expected_tasks,
        "expected_task_ids": expected_task_ids,
        "required_eeg_channels": required,
        "total_subject_session_task_records": len(records),
        "complete_multimodal_records": len(complete_records),
        "analysis_eligible_records": len(eligible_records),
        "subjects_with_at_least_one_complete_record": len(
            subjects_with_complete
        ),
        "complete_records_missing_required_channels": len(
            incompatible_complete
        ),
        "primary_complete_subject_count": len(primary_subjects),
        "primary_complete_record_count": len(primary_records),
        "primary_complete_subjects": primary_subjects,
        "complete_record_distribution": _distribution(records),
    }

    return records, summary


def write_bbbd_manifest(
    records: list[dict[str, Any]],
    summary: Mapping[str, Any],
    output_directory: str | Path,
) -> tuple[Path, Path]:
    """Write a recording-level CSV manifest and JSON summary."""

    output_directory = Path(output_directory).resolve(strict=False)
    output_directory.mkdir(parents=True, exist_ok=True)

    dataset_name = str(summary["dataset"])
    csv_path = output_directory / f"{dataset_name}_records.csv"
    json_path = output_directory / f"{dataset_name}_summary.json"

    fieldnames = [
        "dataset",
        "subject",
        "session",
        "task",
        "label",
        "complete_multimodal",
        "required_channels_available",
        "analysis_eligible",
        "primary_cohort",
        "available_eeg_channel_count",
        "missing_required_channels",
        "eeg_path",
        "channels_path",
        "ecg_path",
        "pupil_path",
        "events_path",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(records)

    json_path.write_text(
        json.dumps(dict(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    return csv_path, json_path


def _run_dataset(
    config: Mapping[str, Any],
    dataset_key: str,
) -> dict[str, Any]:
    dataset_config = config["datasets"][dataset_key]
    archive_path = config["paths"][f"{dataset_key}_archive"]

    records, summary = scan_bbbd_archive(
        archive_path,
        dataset_name=dataset_key,
        expected_tasks=int(dataset_config["expected_tasks"]),
        required_channels=config["eeg"]["internal_channels"],
        label_map=dataset_config["label_map"],
    )

    manifest_directory = (
        Path(config["paths"]["artifacts"]) / "manifests"
    )

    csv_path, json_path = write_bbbd_manifest(
        records,
        summary,
        manifest_directory,
    )

    print(f"\n===== {dataset_key} =====")
    print(
        "All subject-session-task records:",
        summary["total_subject_session_task_records"],
    )
    print(
        "Complete EEG+ECG+Pupil records:",
        summary["complete_multimodal_records"],
    )
    print(
        "Analysis-eligible records:",
        summary["analysis_eligible_records"],
    )
    print(
        "Complete records missing required channels:",
        summary["complete_records_missing_required_channels"],
    )
    print(
        "Primary complete subjects:",
        summary["primary_complete_subject_count"],
    )
    print(
        "Primary complete records:",
        summary["primary_complete_record_count"],
    )
    print(
        "Primary subjects:",
        ", ".join(summary["primary_complete_subjects"]),
    )
    print("CSV manifest:", csv_path)
    print("JSON summary:", json_path)

    return summary


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Build read-only BBBD recording manifests."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=[
            "bbbd_experiment2",
            "bbbd_experiment3",
        ],
        choices=[
            "bbbd_experiment2",
            "bbbd_experiment3",
        ],
    )

    args = parser.parse_args()
    config = load_revision_config(check_input_paths=True)

    for dataset_key in args.datasets:
        _run_dataset(config, dataset_key)


if __name__ == "__main__":
    _main()