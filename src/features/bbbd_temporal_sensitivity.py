"""Duration-controlled BBBD feature extraction for temporal sensitivity.

This module is a thin orchestration layer over the locked BBBD raw-signal
extractor. It does not implement alternative preprocessing or feature logic.
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable

import yaml

from src.features.bbbd_features import build_bbbd_features


EXPECTED_DURATIONS = (8, 16, 24, 32)
EXPECTED_STRIDES = (4, 8, 12, 16)
EXPECTED_OVERLAP = 0.5


def resolve_project_path(
    value: str | Path,
) -> Path:
    """Resolve a project-relative path."""

    path = Path(value)

    if path.is_absolute():
        return path

    return Path.cwd() / path


def load_yaml_mapping(
    path: str | Path,
) -> dict[str, Any]:
    """Load a YAML mapping."""

    resolved = resolve_project_path(path)

    if not resolved.is_file():
        raise FileNotFoundError(resolved)

    value = yaml.safe_load(
        resolved.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(value, dict):
        raise RuntimeError(
            f"Expected YAML mapping: {resolved}"
        )

    return value


def duration_directory_name(
    duration_seconds: int,
    overlap_fraction: float,
) -> str:
    """Return a deterministic duration-specific directory name."""

    duration = int(duration_seconds)
    overlap_percent = int(
        round(
            float(overlap_fraction) * 100
        )
    )

    return (
        f"window_{duration:02d}s_"
        f"overlap_{overlap_percent:02d}pct"
    )


def validate_contract(
    primary_config: dict[str, Any],
    contract: dict[str, Any],
) -> list[dict[str, Any]]:
    """Validate the locked sensitivity contract."""

    primary = contract.get(
        "primary_real_time_reference"
    )

    if not isinstance(primary, dict):
        raise RuntimeError(
            "Sensitivity contract lacks primary_real_time_reference."
        )

    if int(primary.get("duration_seconds")) != 4:
        raise RuntimeError(
            "Primary reference is not four seconds."
        )

    if float(primary.get("overlap_fraction")) != EXPECTED_OVERLAP:
        raise RuntimeError(
            "Primary reference overlap is not 50 percent."
        )

    if primary.get(
        "may_be_replaced_by_sensitivity"
    ) is not False:
        raise RuntimeError(
            "Primary reference is not explicitly immutable."
        )

    primary_windowing = primary_config.get(
        "windowing"
    )

    if not isinstance(primary_windowing, dict):
        raise RuntimeError(
            "Primary BBBD configuration lacks windowing."
        )

    if float(
        primary_windowing.get(
            "duration_seconds"
        )
    ) != 4.0:
        raise RuntimeError(
            "Primary BBBD configuration is not four seconds."
        )

    if float(
        primary_windowing.get(
            "overlap_fraction"
        )
    ) != EXPECTED_OVERLAP:
        raise RuntimeError(
            "Primary BBBD configuration is not 50 percent overlap."
        )

    feature_generation = contract.get(
        "feature_generation"
    )

    if not isinstance(feature_generation, dict):
        raise RuntimeError(
            "Sensitivity contract lacks feature_generation."
        )

    if feature_generation.get(
        "recompute_features_for_each_duration"
    ) is not True:
        raise RuntimeError(
            "Contract does not require raw feature recomputation."
        )

    if feature_generation.get(
        "average_existing_four_second_feature_rows"
    ) is not False:
        raise RuntimeError(
            "Contract improperly permits averaging four-second features."
        )

    if feature_generation.get(
        "concatenate_existing_feature_vectors"
    ) is not False:
        raise RuntimeError(
            "Contract improperly permits concatenating four-second features."
        )

    windows = contract.get(
        "sensitivity_windows"
    )

    if not isinstance(windows, list):
        raise RuntimeError(
            "Sensitivity contract lacks sensitivity_windows."
        )

    observed_durations = [
        int(row["duration_seconds"])
        for row in windows
    ]

    observed_strides = [
        int(row["stride_seconds"])
        for row in windows
    ]

    observed_overlaps = [
        float(row["overlap_fraction"])
        for row in windows
    ]

    if observed_durations != list(EXPECTED_DURATIONS):
        raise RuntimeError(
            f"Unexpected sensitivity durations: {observed_durations}"
        )

    if observed_strides != list(EXPECTED_STRIDES):
        raise RuntimeError(
            f"Unexpected sensitivity strides: {observed_strides}"
        )

    if any(
        overlap != EXPECTED_OVERLAP
        for overlap in observed_overlaps
    ):
        raise RuntimeError(
            "Every sensitivity duration must use 50 percent overlap."
        )

    return windows


def build_duration_config(
    primary_config: dict[str, Any],
    duration_seconds: int,
    overlap_fraction: float,
    output_directory: str | Path,
) -> dict[str, Any]:
    """Create one duration-specific configuration."""

    config = copy.deepcopy(
        primary_config
    )

    config["windowing"][
        "duration_seconds"
    ] = float(duration_seconds)

    config["windowing"][
        "overlap_fraction"
    ] = float(overlap_fraction)

    config["outputs"][
        "full_directory"
    ] = str(output_directory)

    config["outputs"][
        "smoke_directory"
    ] = str(output_directory)

    return config


def validate_duration_summary(
    summary_path: Path,
    duration_seconds: int,
    overlap_fraction: float,
    smoke: bool,
) -> dict[str, Any]:
    """Validate one extracted duration summary."""

    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)

    summary = json.loads(
        summary_path.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(summary, dict):
        raise RuntimeError(
            f"Expected summary mapping: {summary_path}"
        )

    if float(
        summary.get(
            "window_duration_seconds"
        )
    ) != float(duration_seconds):
        raise RuntimeError(
            f"Duration mismatch in {summary_path}"
        )

    if float(
        summary.get(
            "window_overlap_fraction"
        )
    ) != float(overlap_fraction):
        raise RuntimeError(
            f"Overlap mismatch in {summary_path}"
        )

    expected_records = (
        6
        if smoke
        else 392
    )

    if int(
        summary.get(
            "requested_recordings"
        )
    ) != expected_records:
        raise RuntimeError(
            "Unexpected requested-recording count for "
            f"{duration_seconds} seconds."
        )

    if int(
        summary.get(
            "completed_recordings"
        )
    ) != expected_records:
        raise RuntimeError(
            "Unexpected completed-recording count for "
            f"{duration_seconds} seconds."
        )

    if int(
        summary.get(
            "candidate_windows",
            0,
        )
    ) <= 0:
        raise RuntimeError(
            f"No candidate windows for {duration_seconds} seconds."
        )

    if int(
        summary.get(
            "accepted_windows",
            0,
        )
    ) <= 0:
        raise RuntimeError(
            f"No accepted windows for {duration_seconds} seconds."
        )

    if int(
        summary.get(
            "rejected_nonfinite_feature_windows",
            -1,
        )
    ) != 0:
        raise RuntimeError(
            "Nonfinite feature windows were produced for "
            f"{duration_seconds} seconds."
        )

    if summary.get(
        "feature_selection_performed"
    ) is not False:
        raise RuntimeError(
            "Feature selection was unexpectedly performed."
        )

    return summary


def run_temporal_sensitivity_features(
    primary_config_path: str | Path,
    contract_path: str | Path,
    output_root: str | Path,
    *,
    smoke: bool = False,
    durations: Iterable[int] | None = None,
) -> dict[str, Any]:
    """Extract raw BBBD features for the locked duration grid."""

    primary_config = load_yaml_mapping(
        primary_config_path
    )

    contract = load_yaml_mapping(
        contract_path
    )

    windows = validate_contract(
        primary_config,
        contract,
    )

    configured = {
        int(row["duration_seconds"]):
            row
        for row in windows
    }

    selected_durations = (
        list(EXPECTED_DURATIONS)
        if durations is None
        else [
            int(value)
            for value in durations
        ]
    )

    if not selected_durations:
        raise ValueError(
            "At least one duration is required."
        )

    if len(
        selected_durations
    ) != len(
        set(
            selected_durations
        )
    ):
        raise ValueError(
            "Duplicate durations are not allowed."
        )

    unknown = sorted(
        set(selected_durations)
        - set(configured)
    )

    if unknown:
        raise ValueError(
            f"Durations are outside the locked grid: {unknown}"
        )

    resolved_output_root = resolve_project_path(
        output_root
    )

    partial_root = resolved_output_root.with_name(
        resolved_output_root.name
        + ".partial"
    )

    if resolved_output_root.exists():
        raise FileExistsError(
            resolved_output_root
        )

    if partial_root.exists():
        raise FileExistsError(
            partial_root
        )

    partial_root.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    partial_root.mkdir(
        parents=False,
        exist_ok=False,
    )

    manifest_rows: list[dict[str, Any]] = []

    try:
        with tempfile.TemporaryDirectory(
            prefix="bbbd_temporal_configs_"
        ) as temporary_directory:
            temporary_root = Path(
                temporary_directory
            )

            for duration in selected_durations:
                row = configured[
                    duration
                ]

                overlap = float(
                    row[
                        "overlap_fraction"
                    ]
                )

                stride = int(
                    row[
                        "stride_seconds"
                    ]
                )

                expected_stride = int(
                    round(
                        duration
                        * (
                            1.0
                            - overlap
                        )
                    )
                )

                if stride != expected_stride:
                    raise RuntimeError(
                        "Configured stride does not match duration "
                        f"and overlap for {duration} seconds."
                    )

                directory_name = duration_directory_name(
                    duration,
                    overlap,
                )

                duration_output = (
                    partial_root
                    / directory_name
                )

                duration_config = build_duration_config(
                    primary_config,
                    duration,
                    overlap,
                    duration_output,
                )

                temporary_config = (
                    temporary_root
                    / f"bbbd_{duration:02d}s.yaml"
                )

                temporary_config.write_text(
                    yaml.safe_dump(
                        duration_config,
                        sort_keys=False,
                        allow_unicode=False,
                    ),
                    encoding="utf-8",
                )

                build_bbbd_features(
                    config_path=
                        temporary_config,
                    output_directory=
                        duration_output,
                    smoke=
                        smoke,
                )

                summary = validate_duration_summary(
                    duration_output
                    / "summary.json",
                    duration,
                    overlap,
                    smoke,
                )

                manifest_rows.append(
                    {
                        "duration_seconds":
                            duration,
                        "overlap_fraction":
                            overlap,
                        "stride_seconds":
                            stride,
                        "directory":
                            directory_name,
                        "requested_recordings":
                            int(
                                summary[
                                    "requested_recordings"
                                ]
                            ),
                        "completed_recordings":
                            int(
                                summary[
                                    "completed_recordings"
                                ]
                            ),
                        "candidate_windows":
                            int(
                                summary[
                                    "candidate_windows"
                                ]
                            ),
                        "accepted_windows":
                            int(
                                summary[
                                    "accepted_windows"
                                ]
                            ),
                        "rejected_pupil_windows":
                            int(
                                summary[
                                    "rejected_pupil_windows"
                                ]
                            ),
                        "rejected_nonfinite_feature_windows":
                            int(
                                summary[
                                    "rejected_nonfinite_feature_windows"
                                ]
                            ),
                    }
                )

        manifest = {
            "identity":
                "bbbd_temporal_sensitivity_features_v1",
            "primary_config":
                str(
                    primary_config_path
                ),
            "contract":
                str(
                    contract_path
                ),
            "smoke":
                bool(smoke),
            "performance_evaluation_performed":
                False,
            "feature_selection_performed":
                False,
            "model_fitting_performed":
                False,
            "durations":
                manifest_rows,
        }

        (
            partial_root
            / "temporal_sensitivity_manifest.json"
        ).write_text(
            json.dumps(
                manifest,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        partial_root.replace(
            resolved_output_root
        )

        return manifest

    except Exception:
        if partial_root.exists():
            shutil.rmtree(
                partial_root
            )

        raise


def main() -> None:
    """Command-line entry point."""

    parser = argparse.ArgumentParser(
        description=(
            "Extract raw BBBD features for the locked "
            "temporal-sensitivity window grid."
        )
    )

    parser.add_argument(
        "--primary-config",
        default="configs/bbbd.yaml",
    )

    parser.add_argument(
        "--contract",
        default="configs/bbbd_temporal_sensitivity.yaml",
    )

    parser.add_argument(
        "--output-root",
        required=True,
    )

    parser.add_argument(
        "--smoke",
        action="store_true",
    )

    parser.add_argument(
        "--durations",
        nargs="*",
        type=int,
        default=None,
    )

    arguments = parser.parse_args()

    result = run_temporal_sensitivity_features(
        primary_config_path=
            arguments.primary_config,
        contract_path=
            arguments.contract,
        output_root=
            arguments.output_root,
        smoke=
            arguments.smoke,
        durations=
            arguments.durations,
    )

    print(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
