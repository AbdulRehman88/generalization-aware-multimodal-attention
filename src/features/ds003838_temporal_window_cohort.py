"""Build prespecified ds003838 temporal-window feature cohorts."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.core.config import load_revision_config
from src.features.ds003838_revision_features import (
    extract_subject_tables,
)
from src.features.ds003838_temporal_windows import (
    EXPECTED_WINDOW_NAMES,
    build_candidate_config,
    temporal_window_protocol,
)
from src.features.ds003838_trial_relative_features import (
    build_baseline_config,
    build_delta_table,
    parse_boolean_series,
)
from src.features.internal_xr_revision_features import (
    feature_columns,
)
from src.preprocessing.ds003838_cohort import (
    verify_subject_output,
)
from src.preprocessing.ds003838_revision import (
    preprocess_subject,
    write_subject_output,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

EXPECTED_PARTICIPANTS = 58
EXPECTED_ROWS = 6264
EXPECTED_ROWS_PER_PARTICIPANT = 108
EXPECTED_FEATURES = 219
EXPECTED_LABEL_COUNTS = {
    0: 2088,
    1: 2088,
    2: 2088,
}

ALIGNMENT_COLUMNS = [
    "segment_id",
    "participant",
    "phase",
    "label",
    "window_index",
]


def resolve_project_path(
    value: str | Path,
) -> Path:
    """Resolve one project-relative path."""

    path = Path(value)

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path.resolve()


def sha256_file(
    path: Path,
) -> str:
    """Compute the SHA-256 digest of one file."""

    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for block in iter(
            lambda: stream.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def validate_complete_table(
    table: pd.DataFrame,
    *,
    table_name: str,
) -> list[str]:
    """Validate one complete temporal-window aggregate."""

    features = feature_columns(
        table
    )

    if len(table) != EXPECTED_ROWS:
        raise RuntimeError(
            f"{table_name}: expected {EXPECTED_ROWS} rows, "
            f"observed {len(table)}."
        )

    if (
        table["participant"].astype(str).nunique()
        != EXPECTED_PARTICIPANTS
    ):
        raise RuntimeError(
            f"{table_name}: expected "
            f"{EXPECTED_PARTICIPANTS} participants."
        )

    if len(features) != EXPECTED_FEATURES:
        raise RuntimeError(
            f"{table_name}: expected {EXPECTED_FEATURES} features, "
            f"observed {len(features)}."
        )

    if (
        table["label"]
        .astype(int)
        .value_counts()
        .sort_index()
        .to_dict()
        != EXPECTED_LABEL_COUNTS
    ):
        raise RuntimeError(
            f"{table_name}: class counts changed."
        )

    if table["segment_id"].duplicated().any():
        raise RuntimeError(
            f"{table_name}: duplicate segment identifiers."
        )

    participant_counts = (
        table.groupby("participant")
        .size()
    )

    if not (
        participant_counts
        == EXPECTED_ROWS_PER_PARTICIPANT
    ).all():
        raise RuntimeError(
            f"{table_name}: every participant must contribute "
            f"{EXPECTED_ROWS_PER_PARTICIPANT} rows."
        )

    participant_label_counts = (
        table.groupby(
            [
                "participant",
                "label",
            ]
        )
        .size()
        .unstack(fill_value=0)
    )

    if participant_label_counts.shape != (
        EXPECTED_PARTICIPANTS,
        3,
    ):
        raise RuntimeError(
            f"{table_name}: participant-class matrix changed."
        )

    if not (
        participant_label_counts == 36
    ).all().all():
        raise RuntimeError(
            f"{table_name}: every participant must contribute "
            "36 rows per class."
        )

    values = table[
        features
    ].to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise RuntimeError(
            f"{table_name}: nonfinite values detected."
        )

    return features


def validate_window_alignment(
    tables: dict[str, pd.DataFrame],
) -> None:
    """Require identical ordered trial metadata across windows."""

    if list(tables) != EXPECTED_WINDOW_NAMES:
        raise RuntimeError(
            f"Window registry changed: {list(tables)}"
        )

    reference_name = EXPECTED_WINDOW_NAMES[0]

    reference = (
        tables[reference_name]
        .sort_values("segment_id")
        .reset_index(drop=True)
    )

    for window_name in EXPECTED_WINDOW_NAMES[1:]:
        candidate = (
            tables[window_name]
            .sort_values("segment_id")
            .reset_index(drop=True)
        )

        for column in ALIGNMENT_COLUMNS:
            left = reference[
                column
            ].astype(str)

            right = candidate[
                column
            ].astype(str)

            if not left.equals(right):
                raise RuntimeError(
                    f"{window_name}: trial alignment differs "
                    f"for metadata column {column}."
                )


def write_subject_shard(
    table: pd.DataFrame,
    *,
    subject: str,
    candidate: dict[str, Any],
    destination: Path,
) -> None:
    """Write one temporal-window feature shard atomically."""

    if destination.exists():
        raise FileExistsError(
            f"Refusing to overwrite shard: {destination}"
        )

    temporary = destination.with_name(
        destination.name + ".building"
    )

    if temporary.exists():
        raise FileExistsError(
            f"Temporary shard already exists: {temporary}"
        )

    temporary.mkdir(
        parents=True,
        exist_ok=False,
    )

    features = feature_columns(
        table
    )

    values = table[
        features
    ].to_numpy(dtype=float)

    table_path = (
        temporary
        / "ECG_EEG_delta_features.csv"
    )

    table.to_csv(
        table_path,
        index=False,
    )

    summary = {
        "dataset": "ds003838",
        "subject": subject,
        "window": str(
            candidate["name"]
        ),
        "anchor": str(
            candidate["anchor"]
        ),
        "start_offset_seconds": float(
            candidate[
                "start_offset_seconds"
            ]
        ),
        "duration_seconds": float(
            candidate[
                "duration_seconds"
            ]
        ),
        "representation":
            "post_window_minus_pretrial_baseline",
        "rows": int(len(table)),
        "feature_count": int(len(features)),
        "label_counts": {
            str(key): int(value)
            for key, value
            in table["label"]
            .astype(int)
            .value_counts()
            .sort_index()
            .to_dict()
            .items()
        },
        "all_values_finite": bool(
            np.isfinite(values).all()
        ),
        "median_absolute_delta": float(
            np.median(
                np.abs(values)
            )
        ),
    }

    (
        temporary / "summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )

    temporary.replace(
        destination
    )


def verify_subject_shard(
    shard_directory: Path,
    *,
    subject: str,
    candidate_name: str,
) -> dict[str, Any]:
    """Verify one completed temporal-window feature shard."""

    table_path = (
        shard_directory
        / "ECG_EEG_delta_features.csv"
    )

    summary_path = (
        shard_directory
        / "summary.json"
    )

    if not table_path.is_file():
        raise RuntimeError(
            f"{subject}/{candidate_name}: feature table missing."
        )

    if not summary_path.is_file():
        raise RuntimeError(
            f"{subject}/{candidate_name}: summary missing."
        )

    table = pd.read_csv(
        table_path,
        low_memory=False,
    )

    summary = json.loads(
        summary_path.read_text(
            encoding="utf-8"
        )
    )

    features = feature_columns(
        table
    )

    if len(table) != EXPECTED_ROWS_PER_PARTICIPANT:
        raise RuntimeError(
            f"{subject}/{candidate_name}: expected "
            f"{EXPECTED_ROWS_PER_PARTICIPANT} rows."
        )

    if len(features) != EXPECTED_FEATURES:
        raise RuntimeError(
            f"{subject}/{candidate_name}: expected "
            f"{EXPECTED_FEATURES} features."
        )

    if set(
        table["participant"].astype(str)
    ) != {
        subject
    }:
        raise RuntimeError(
            f"{subject}/{candidate_name}: participant changed."
        )

    if (
        table["label"]
        .astype(int)
        .value_counts()
        .sort_index()
        .to_dict()
        != {
            0: 36,
            1: 36,
            2: 36,
        }
    ):
        raise RuntimeError(
            f"{subject}/{candidate_name}: labels changed."
        )

    if table["segment_id"].duplicated().any():
        raise RuntimeError(
            f"{subject}/{candidate_name}: duplicate segments."
        )

    values = table[
        features
    ].to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise RuntimeError(
            f"{subject}/{candidate_name}: nonfinite values."
        )

    if str(
        summary["window"]
    ) != candidate_name:
        raise RuntimeError(
            f"{subject}/{candidate_name}: summary window changed."
        )

    return {
        "subject": subject,
        "window": candidate_name,
        "rows": int(len(table)),
        "features": int(len(features)),
        "median_absolute_delta": float(
            np.median(
                np.abs(values)
            )
        ),
    }


def aggregate_subject_shards(
    *,
    candidate_name: str,
    participant_order: list[str],
    shard_root: Path,
    destination: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Aggregate and verify one complete candidate cohort."""

    frames = []

    for subject in participant_order:
        table = pd.read_csv(
            shard_root
            / subject
            / "ECG_EEG_delta_features.csv",
            low_memory=False,
        )

        frames.append(
            table
        )

    aggregate = pd.concat(
        frames,
        ignore_index=True,
    )

    features = validate_complete_table(
        aggregate,
        table_name=candidate_name,
    )

    aggregate_path = (
        destination
        / "ECG_EEG_delta_features.csv"
    )

    aggregate.to_csv(
        aggregate_path,
        index=False,
    )

    values = aggregate[
        features
    ].to_numpy(dtype=float)

    candidate_summary = {
        "window": candidate_name,
        "rows": int(len(aggregate)),
        "participants": int(
            aggregate[
                "participant"
            ].nunique()
        ),
        "features": int(len(features)),
        "label_counts": {
            str(key): int(value)
            for key, value
            in aggregate["label"]
            .astype(int)
            .value_counts()
            .sort_index()
            .to_dict()
            .items()
        },
        "median_absolute_delta": float(
            np.median(
                np.abs(values)
            )
        ),
        "aggregate_path": str(
            aggregate_path.relative_to(
                PROJECT_ROOT
            )
        ).replace("\\", "/"),
        "aggregate_sha256":
            sha256_file(
                aggregate_path
            ),
    }

    return aggregate, candidate_summary


def build_candidate_cohort(
    *,
    config: dict[str, Any],
    baseline_config: dict[str, Any],
    strict_manifest: pd.DataFrame,
    candidate: dict[str, Any],
    post_root: Path,
    baseline_root: Path,
    output_root: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build or resume one complete temporal-window cohort."""

    candidate_name = str(
        candidate["name"]
    )

    candidate_config = build_candidate_config(
        config,
        candidate,
    )

    candidate_post_root = (
        post_root / candidate_name
    )

    candidate_output = (
        output_root / candidate_name
    )

    candidate_post_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    candidate_output.mkdir(
        parents=True,
        exist_ok=True,
    )

    shard_root = (
        candidate_output
        / "subjects"
    )

    shard_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    shard_rows = []

    total = len(
        strict_manifest
    )

    for index, manifest_row in (
        strict_manifest.iterrows()
    ):
        subject = str(
            manifest_row["subject"]
        )

        print(
            f"[{candidate_name}] "
            f"[{index + 1:02d}/{total:02d}] "
            f"{subject}",
            flush=True,
        )

        baseline_subject = (
            baseline_root / subject
        )

        verify_subject_output(
            baseline_subject,
            expected_subject=subject,
            expected_channels=7,
            expected_trials=
                EXPECTED_ROWS_PER_PARTICIPANT,
        )

        post_subject = (
            candidate_post_root
            / subject
        )

        if not post_subject.exists():
            print(
                "  Preprocessing candidate window.",
                flush=True,
            )

            arrays, metadata, audit = (
                preprocess_subject(
                    manifest_row,
                    candidate_config,
                )
            )

            write_subject_output(
                arrays,
                metadata,
                audit,
                post_subject,
            )

        else:
            print(
                "  Existing candidate window found; verifying.",
                flush=True,
            )

        verify_subject_output(
            post_subject,
            expected_subject=subject,
            expected_channels=7,
            expected_trials=
                EXPECTED_ROWS_PER_PARTICIPANT,
        )

        shard_directory = (
            shard_root / subject
        )

        if not shard_directory.exists():
            print(
                "  Extracting trial-relative features.",
                flush=True,
            )

            baseline_table = (
                extract_subject_tables(
                    baseline_subject,
                    baseline_config,
                )[
                    "ECG_EEG"
                ]
            )

            post_table = (
                extract_subject_tables(
                    post_subject,
                    candidate_config,
                )[
                    "ECG_EEG"
                ]
            )

            delta = build_delta_table(
                post_table,
                baseline_table,
            )

            write_subject_shard(
                delta,
                subject=subject,
                candidate=candidate,
                destination=
                    shard_directory,
            )

        else:
            print(
                "  Existing feature shard found; verifying.",
                flush=True,
            )

        verified = verify_subject_shard(
            shard_directory,
            subject=subject,
            candidate_name=
                candidate_name,
        )

        shard_rows.append(
            verified
        )

        print(
            "  Verified: "
            f"{verified['rows']} rows, "
            f"{verified['features']} features.",
            flush=True,
        )

    pd.DataFrame(
        shard_rows
    ).to_csv(
        candidate_output
        / "cohort_subjects.csv",
        index=False,
    )

    participant_order = (
        strict_manifest[
            "subject"
        ].astype(str).tolist()
    )

    aggregate, candidate_summary = (
        aggregate_subject_shards(
            candidate_name=
                candidate_name,
            participant_order=
                participant_order,
            shard_root=shard_root,
            destination=
                candidate_output,
        )
    )

    candidate_summary.update(
        {
            "anchor": str(
                candidate["anchor"]
            ),
            "start_offset_seconds":
                float(
                    candidate[
                        "start_offset_seconds"
                    ]
                ),
            "duration_seconds":
                float(
                    candidate[
                        "duration_seconds"
                    ]
                ),
        }
    )

    (
        candidate_output
        / "cohort_summary.json"
    ).write_text(
        json.dumps(
            candidate_summary,
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )

    return aggregate, candidate_summary


def reuse_delayed_retention(
    *,
    candidate: dict[str, Any],
    output_root: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Reuse the previously verified delayed-retention cohort."""

    source_root = resolve_project_path(
        "outputs/revision/features/"
        "ds003838_trial_relative"
    )

    source_table = (
        source_root
        / "ECG_EEG_delta_features.csv"
    )

    if not source_table.is_file():
        raise RuntimeError(
            "Locked delayed-retention aggregate is missing."
        )

    destination = (
        output_root
        / "delayed_retention"
    )

    destination.mkdir(
        parents=True,
        exist_ok=True,
    )

    destination_table = (
        destination
        / "ECG_EEG_delta_features.csv"
    )

    if not destination_table.exists():
        shutil.copy2(
            source_table,
            destination_table,
        )

    table = pd.read_csv(
        destination_table,
        low_memory=False,
    )

    features = validate_complete_table(
        table,
        table_name=
            "delayed_retention",
    )

    values = table[
        features
    ].to_numpy(dtype=float)

    summary = {
        "window":
            "delayed_retention",
        "anchor": str(
            candidate["anchor"]
        ),
        "start_offset_seconds": float(
            candidate[
                "start_offset_seconds"
            ]
        ),
        "duration_seconds": float(
            candidate[
                "duration_seconds"
            ]
        ),
        "rows": int(len(table)),
        "participants": int(
            table[
                "participant"
            ].nunique()
        ),
        "features": int(len(features)),
        "label_counts": {
            str(key): int(value)
            for key, value
            in table["label"]
            .astype(int)
            .value_counts()
            .sort_index()
            .to_dict()
            .items()
        },
        "median_absolute_delta": float(
            np.median(
                np.abs(values)
            )
        ),
        "aggregate_path": str(
            destination_table.relative_to(
                PROJECT_ROOT
            )
        ).replace("\\", "/"),
        "aggregate_sha256":
            sha256_file(
                destination_table
            ),
        "reused_from": str(
            source_table.relative_to(
                PROJECT_ROOT
            )
        ).replace("\\", "/"),
    }

    (
        destination
        / "cohort_summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )

    return table, summary


def run_temporal_window_cohorts(
    config: dict[str, Any],
    *,
    post_root: str | Path,
    baseline_root: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    """Build all candidate cohorts without selecting a window."""

    protocol = temporal_window_protocol(
        config
    )

    candidates = {
        str(candidate["name"]):
            candidate
        for candidate in protocol[
            "candidates"
        ]
    }

    if list(candidates) != EXPECTED_WINDOW_NAMES:
        raise RuntimeError(
            "Temporal-window candidate order changed."
        )

    manifest = pd.read_csv(
        resolve_project_path(
            "outputs/revision/manifests/"
            "ds003838/ds003838_subjects.csv"
        ),
        low_memory=False,
    )

    strict_mask = parse_boolean_series(
        manifest[
            "strict_primary_cohort"
        ]
    )

    strict = (
        manifest.loc[
            strict_mask
        ]
        .sort_values("subject")
        .reset_index(drop=True)
    )

    if len(strict) != EXPECTED_PARTICIPANTS:
        raise RuntimeError(
            f"Expected {EXPECTED_PARTICIPANTS} participants, "
            f"observed {len(strict)}."
        )

    destination = resolve_project_path(
        output_root
    )

    destination.mkdir(
        parents=True,
        exist_ok=True,
    )

    post_directory = resolve_project_path(
        post_root
    )

    baseline_directory = resolve_project_path(
        baseline_root
    )

    baseline_config = build_baseline_config(
        config
    )

    tables: dict[str, pd.DataFrame] = {}
    summaries: dict[str, Any] = {}

    for window_name in [
        "late_encoding",
        "immediate_retention",
    ]:
        print(
            f"\n===== BUILD {window_name.upper()} =====",
            flush=True,
        )

        table, candidate_summary = (
            build_candidate_cohort(
                config=config,
                baseline_config=
                    baseline_config,
                strict_manifest=strict,
                candidate=
                    candidates[
                        window_name
                    ],
                post_root=
                    post_directory,
                baseline_root=
                    baseline_directory,
                output_root=
                    destination,
            )
        )

        tables[window_name] = table
        summaries[window_name] = (
            candidate_summary
        )

    print(
        "\n===== REUSE LOCKED DELAYED RETENTION =====",
        flush=True,
    )

    delayed_table, delayed_summary = (
        reuse_delayed_retention(
            candidate=
                candidates[
                    "delayed_retention"
                ],
            output_root=
                destination,
        )
    )

    tables[
        "delayed_retention"
    ] = delayed_table

    summaries[
        "delayed_retention"
    ] = delayed_summary

    ordered_tables = {
        name: tables[name]
        for name in EXPECTED_WINDOW_NAMES
    }

    validate_window_alignment(
        ordered_tables
    )

    summary = {
        "dataset": "ds003838",
        "cohort": "strict_primary",
        "representation":
            "post_window_minus_pretrial_baseline",
        "participant_count":
            EXPECTED_PARTICIPANTS,
        "rows_per_window":
            EXPECTED_ROWS,
        "features_per_window":
            EXPECTED_FEATURES,
        "candidate_order":
            EXPECTED_WINDOW_NAMES,
        "baseline": protocol[
            "baseline"
        ],
        "windows": {
            name: summaries[name]
            for name
            in EXPECTED_WINDOW_NAMES
        },
        "trial_metadata_aligned_across_windows":
            True,
        "classification_performed":
            False,
        "global_window_selection_performed":
            False,
        "permitted_future_selection_scope":
            "participant_disjoint_inner_validation_only",
    }

    (
        destination
        / "temporal_window_cohort_summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )

    print(
        "\n===== COMPLETE TEMPORAL-WINDOW COHORT SUMMARY ====="
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
            "Build all complete prespecified "
            "ds003838 temporal-window cohorts."
        )
    )

    parser.add_argument(
        "--post-root",
        default=(
            "outputs/revision/preprocessed/"
            "ds003838_temporal_candidates"
        ),
    )

    parser.add_argument(
        "--baseline-root",
        default=(
            "outputs/revision/preprocessed/"
            "ds003838_trial_baseline"
        ),
    )

    parser.add_argument(
        "--output-root",
        default=(
            "outputs/revision/features/"
            "ds003838_temporal_candidates"
        ),
    )

    arguments = parser.parse_args()

    config = load_revision_config(
        check_input_paths=True
    )

    run_temporal_window_cohorts(
        config,
        post_root=arguments.post_root,
        baseline_root=
            arguments.baseline_root,
        output_root=
            arguments.output_root,
    )


if __name__ == "__main__":
    main()