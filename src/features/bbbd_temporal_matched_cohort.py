"""Materialize the matched BBBD temporal-sensitivity feature cohort.

The module filters already extracted duration-specific feature tables using
the locked recording registry. It does not recompute physiological features,
fit selectors, train classifiers, or inspect predictive performance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Mapping

import pandas as pd


FEATURE_FILES = {
    "EEG": "EEG_features.csv",
    "ECG": "ECG_features.csv",
    "Pupil": "Pupil_features.csv",
    "ECG_EEG": "ECG_EEG_features.csv",
    "ECG_Pupil": "ECG_Pupil_features.csv",
    "EEG_Pupil": "EEG_Pupil_features.csv",
    "ECG_EEG_Pupil": "ECG_EEG_Pupil_features.csv",
}

DEFAULT_DURATION_SOURCES = {
    4: "outputs/revision/features/bbbd_primary",
    8: (
        "outputs/revision/features/bbbd_temporal_sensitivity_v1/"
        "window_08s_overlap_50pct"
    ),
    16: (
        "outputs/revision/features/bbbd_temporal_sensitivity_v1/"
        "window_16s_overlap_50pct"
    ),
    24: (
        "outputs/revision/features/bbbd_temporal_sensitivity_v1/"
        "window_24s_overlap_50pct"
    ),
    32: (
        "outputs/revision/features/bbbd_temporal_sensitivity_v1/"
        "window_32s_overlap_50pct"
    ),
}

EXPECTED_DURATIONS = (4, 8, 16, 24, 32)
EXPECTED_RECORDINGS = 387
EXPECTED_PARTICIPANTS = 36


def resolve_project_path(value: str | Path) -> Path:
    """Resolve a project-relative path."""

    path = Path(value)

    if path.is_absolute():
        return path

    return Path.cwd() / path


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one file."""

    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for block in iter(
            lambda: stream.read(4 * 1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def boolean_mask(values: pd.Series) -> pd.Series:
    """Parse a strict Boolean registry-selection column."""

    if pd.api.types.is_bool_dtype(values):
        return values.astype(bool)

    normalized = (
        values.astype(str)
        .str.strip()
        .str.lower()
    )

    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
    }

    unknown = sorted(
        set(normalized)
        - set(mapping)
    )

    if unknown:
        raise ValueError(
            f"Unknown Boolean registry values: {unknown}"
        )

    return normalized.map(mapping).astype(bool)


def duration_directory_name(duration_seconds: int) -> str:
    """Return the immutable matched duration-directory name."""

    duration = int(duration_seconds)

    if duration not in EXPECTED_DURATIONS:
        raise ValueError(
            f"Duration is outside the locked grid: {duration}"
        )

    return f"window_{duration:02d}s_matched_387"


def load_selected_registry(
    registry_path: str | Path,
    selection_column: str,
) -> pd.DataFrame:
    """Load and validate the objective matched-recording registry."""

    resolved = resolve_project_path(registry_path)

    if not resolved.is_file():
        raise FileNotFoundError(resolved)

    registry = pd.read_csv(
        resolved,
        low_memory=False,
    )

    required = {
        "recording_id",
        "dataset",
        "participant",
        "label",
        selection_column,
    }

    missing = required - set(registry.columns)

    if missing:
        raise RuntimeError(
            f"Registry lacks columns: {sorted(missing)}"
        )

    if registry["recording_id"].duplicated().any():
        raise RuntimeError(
            "Registry contains duplicate recording IDs."
        )

    mask = boolean_mask(
        registry[selection_column]
    )

    selected = registry.loc[
        mask
    ].copy()

    selected["recording_id"] = selected[
        "recording_id"
    ].astype(str)

    if len(selected) != EXPECTED_RECORDINGS:
        raise RuntimeError(
            "Matched registry does not select exactly "
            f"{EXPECTED_RECORDINGS} recordings."
        )

    if selected["participant"].nunique() != EXPECTED_PARTICIPANTS:
        raise RuntimeError(
            "Matched registry does not retain all 36 participants."
        )

    if set(selected["dataset"].astype(str)) != {
        "experiment2",
        "experiment3",
    }:
        raise RuntimeError(
            "Matched registry does not retain both experiments."
        )

    if set(selected["label"].astype(int)) != {
        0,
        1,
    }:
        raise RuntimeError(
            "Matched registry does not retain both labels."
        )

    return selected


def materialize_matched_features(
    registry_path: str | Path,
    output_root: str | Path,
    *,
    selection_column: str = "temporal_common_complete_case",
    duration_sources: Mapping[int, str | Path] | None = None,
) -> dict[str, Any]:
    """Filter all duration feature tables to the matched registry."""

    sources = (
        dict(DEFAULT_DURATION_SOURCES)
        if duration_sources is None
        else {
            int(duration): source
            for duration, source in duration_sources.items()
        }
    )

    if tuple(sorted(sources)) != EXPECTED_DURATIONS:
        raise RuntimeError(
            f"Duration-source grid differs from {EXPECTED_DURATIONS}."
        )

    selected_registry = load_selected_registry(
        registry_path,
        selection_column,
    )

    selected_ids = set(
        selected_registry["recording_id"]
    )

    resolved_output = resolve_project_path(
        output_root
    )

    partial_output = resolved_output.with_name(
        resolved_output.name + ".partial"
    )

    if resolved_output.exists():
        raise FileExistsError(resolved_output)

    if partial_output.exists():
        raise FileExistsError(partial_output)

    partial_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    partial_output.mkdir(
        parents=False,
        exist_ok=False,
    )

    duration_rows: list[dict[str, Any]] = []

    try:
        for duration in EXPECTED_DURATIONS:
            source_root = resolve_project_path(
                sources[duration]
            )

            if not source_root.is_dir():
                raise FileNotFoundError(source_root)

            source_summary_path = (
                source_root / "summary.json"
            )

            source_audit_path = (
                source_root / "recording_audit.csv"
            )

            if not source_summary_path.is_file():
                raise FileNotFoundError(
                    source_summary_path
                )

            if not source_audit_path.is_file():
                raise FileNotFoundError(
                    source_audit_path
                )

            source_summary = json.loads(
                source_summary_path.read_text(
                    encoding="utf-8"
                )
            )

            if not isinstance(source_summary, dict):
                raise RuntimeError(
                    f"Invalid source summary: {source_summary_path}"
                )

            duration_output = (
                partial_output
                / duration_directory_name(duration)
            )

            duration_output.mkdir(
                parents=False,
                exist_ok=False,
            )

            recording_audit = pd.read_csv(
                source_audit_path,
                low_memory=False,
            )

            recording_audit["recording_id"] = recording_audit[
                "recording_id"
            ].astype(str)

            filtered_audit = recording_audit.loc[
                recording_audit["recording_id"].isin(
                    selected_ids
                )
            ].copy()

            if set(filtered_audit["recording_id"]) != selected_ids:
                missing_ids = sorted(
                    selected_ids
                    - set(filtered_audit["recording_id"])
                )

                raise RuntimeError(
                    f"{duration}s recording audit lacks IDs: "
                    f"{missing_ids[:10]}"
                )

            if len(filtered_audit) != EXPECTED_RECORDINGS:
                raise RuntimeError(
                    f"{duration}s filtered audit does not contain "
                    "387 recordings."
                )

            if (
                pd.to_numeric(
                    filtered_audit["accepted_windows"],
                    errors="raise",
                )
                <= 0
            ).any():
                raise RuntimeError(
                    f"{duration}s matched cohort contains a recording "
                    "without an accepted window."
                )

            filtered_audit.to_csv(
                duration_output / "recording_audit.csv",
                index=False,
            )

            reference_segment_ids: list[str] | None = None
            reference_labels: pd.Series | None = None
            row_counts: dict[str, int] = {}
            feature_hashes: dict[str, str] = {}
            schema_hashes: dict[str, str] = {}

            for path_name, file_name in FEATURE_FILES.items():
                source_file = source_root / file_name

                if not source_file.is_file():
                    raise FileNotFoundError(source_file)

                frame = pd.read_csv(
                    source_file,
                    low_memory=False,
                )

                if "recording_id" not in frame.columns:
                    raise RuntimeError(
                        f"{source_file} lacks recording_id."
                    )

                frame["recording_id"] = frame[
                    "recording_id"
                ].astype(str)

                filtered = frame.loc[
                    frame["recording_id"].isin(
                        selected_ids
                    )
                ].copy()

                if filtered.empty:
                    raise RuntimeError(
                        f"{duration}s/{path_name} is empty."
                    )

                if set(filtered["recording_id"]) != selected_ids:
                    raise RuntimeError(
                        f"{duration}s/{path_name} does not contain "
                        "the complete matched registry."
                    )

                if filtered["recording_id"].nunique() != EXPECTED_RECORDINGS:
                    raise RuntimeError(
                        f"{duration}s/{path_name} recording count differs."
                    )

                if filtered["participant"].nunique() != EXPECTED_PARTICIPANTS:
                    raise RuntimeError(
                        f"{duration}s/{path_name} participant count differs."
                    )

                if filtered["segment_id"].duplicated().any():
                    raise RuntimeError(
                        f"{duration}s/{path_name} contains duplicate "
                        "segment IDs."
                    )

                segment_ids = list(
                    filtered["segment_id"].astype(str)
                )

                if reference_segment_ids is None:
                    reference_segment_ids = segment_ids
                    reference_labels = filtered[
                        "label"
                    ].astype(int).reset_index(
                        drop=True
                    )

                elif segment_ids != reference_segment_ids:
                    raise RuntimeError(
                        f"{duration}s seven-path segment alignment failed."
                    )

                elif not filtered[
                    "label"
                ].astype(int).reset_index(
                    drop=True
                ).equals(reference_labels):
                    raise RuntimeError(
                        f"{duration}s seven-path label alignment failed."
                    )

                output_file = duration_output / file_name

                filtered.to_csv(
                    output_file,
                    index=False,
                )

                row_counts[path_name] = int(
                    len(filtered)
                )

                feature_hashes[path_name] = sha256_file(
                    output_file
                )

                schema_hashes[path_name] = hashlib.sha256(
                    "\n".join(
                        filtered.columns.astype(str)
                    ).encode("utf-8")
                ).hexdigest()

            if len(set(row_counts.values())) != 1:
                raise RuntimeError(
                    f"{duration}s path row counts differ: {row_counts}"
                )

            accepted_windows = int(
                pd.to_numeric(
                    filtered_audit["accepted_windows"],
                    errors="raise",
                ).sum()
            )

            candidate_windows = int(
                pd.to_numeric(
                    filtered_audit["candidate_windows"],
                    errors="raise",
                ).sum()
            )

            rejected_pupil = int(
                pd.to_numeric(
                    filtered_audit["rejected_pupil_windows"],
                    errors="raise",
                ).sum()
            )

            rejected_nonfinite = int(
                pd.to_numeric(
                    filtered_audit[
                        "rejected_nonfinite_feature_windows"
                    ],
                    errors="raise",
                ).sum()
            )

            observed_rows = next(
                iter(row_counts.values())
            )

            if observed_rows != accepted_windows:
                raise RuntimeError(
                    f"{duration}s rows differ from accepted-window count."
                )

            base_file = (
                duration_output
                / FEATURE_FILES["ECG_EEG_Pupil"]
            )

            base_metadata = pd.read_csv(
                base_file,
                usecols=[
                    "recording_id",
                    "participant",
                    "label",
                ],
                low_memory=False,
            )

            feature_counts = (
                source_summary.get(
                    "validation",
                    {}
                ).get(
                    "feature_counts"
                )
            )

            if not isinstance(feature_counts, dict):
                raise RuntimeError(
                    f"{duration}s source summary lacks feature counts."
                )

            label_counts = (
                base_metadata["label"]
                .astype(int)
                .value_counts(
                    sort=False
                )
                .sort_index()
                .to_dict()
            )

            matched_summary = {
                "identity":
                    "bbbd_temporal_matched_feature_cohort_v1",
                "dataset":
                    "BBBD",
                "duration_seconds":
                    duration,
                "window_duration_seconds":
                    float(duration),
                "window_overlap_fraction":
                    0.5,
                "source_feature_root":
                    str(sources[duration]),
                "source_summary_sha256":
                    sha256_file(source_summary_path),
                "recording_registry":
                    str(registry_path),
                "registry_selection_column":
                    selection_column,
                "matched_cohort_materialization":
                    True,
                "raw_feature_recomputation_performed":
                    False,
                "feature_selection_performed":
                    False,
                "classification_performed":
                    False,
                "performance_evaluation_performed":
                    False,
                "requested_recordings":
                    EXPECTED_RECORDINGS,
                "completed_recordings":
                    EXPECTED_RECORDINGS,
                "participants":
                    EXPECTED_PARTICIPANTS,
                "candidate_windows":
                    candidate_windows,
                "accepted_windows":
                    accepted_windows,
                "rejected_pupil_windows":
                    rejected_pupil,
                "rejected_nonfinite_feature_windows":
                    rejected_nonfinite,
                "validation": {
                    "participants":
                        int(
                            base_metadata[
                                "participant"
                            ].nunique()
                        ),
                    "recordings":
                        int(
                            base_metadata[
                                "recording_id"
                            ].nunique()
                        ),
                    "rows":
                        int(len(base_metadata)),
                    "label_counts": {
                        str(key): int(value)
                        for key, value
                        in label_counts.items()
                    },
                    "feature_counts":
                        feature_counts,
                },
                "row_counts":
                    row_counts,
                "schema_hashes":
                    schema_hashes,
                "feature_file_sha256":
                    feature_hashes,
            }

            summary_output = (
                duration_output / "summary.json"
            )

            summary_output.write_text(
                json.dumps(
                    matched_summary,
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )

            duration_rows.append(
                {
                    "duration_seconds":
                        duration,
                    "directory":
                        duration_directory_name(duration),
                    "participants":
                        EXPECTED_PARTICIPANTS,
                    "recordings":
                        EXPECTED_RECORDINGS,
                    "candidate_windows":
                        candidate_windows,
                    "accepted_windows":
                        accepted_windows,
                    "rejected_pupil_windows":
                        rejected_pupil,
                    "rejected_nonfinite_feature_windows":
                        rejected_nonfinite,
                    "rows_per_path":
                        observed_rows,
                    "summary_sha256":
                        sha256_file(summary_output),
                }
            )

        manifest = {
            "identity":
                "bbbd_temporal_matched_features_v1",
            "recording_registry":
                str(registry_path),
            "registry_selection_column":
                selection_column,
            "participants":
                EXPECTED_PARTICIPANTS,
            "recordings":
                EXPECTED_RECORDINGS,
            "durations_seconds":
                list(EXPECTED_DURATIONS),
            "feature_selection_performed":
                False,
            "classifier_fitted":
                False,
            "performance_observed":
                False,
            "duration_cohorts":
                duration_rows,
        }

        manifest_path = (
            partial_output / "matched_cohort_manifest.json"
        )

        manifest_path.write_text(
            json.dumps(
                manifest,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        partial_output.replace(
            resolved_output
        )

        return manifest

    except Exception:
        if partial_output.exists():
            shutil.rmtree(partial_output)

        raise


def main() -> None:
    """Command-line entry point."""

    parser = argparse.ArgumentParser(
        description=(
            "Materialize the locked matched BBBD temporal feature cohort."
        )
    )

    parser.add_argument(
        "--registry",
        default=(
            "artifacts/revision/manifests/"
            "bbbd_temporal_sensitivity_recording_registry_v1.csv"
        ),
    )

    parser.add_argument(
        "--selection-column",
        default="temporal_common_complete_case",
    )

    parser.add_argument(
        "--output-root",
        required=True,
    )

    arguments = parser.parse_args()

    manifest = materialize_matched_features(
        registry_path=arguments.registry,
        output_root=arguments.output_root,
        selection_column=arguments.selection_column,
    )

    print(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
