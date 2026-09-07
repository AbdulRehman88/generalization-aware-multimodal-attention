"""Corrected seven-path feature extraction for ds003838."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.core.config import load_revision_config
from src.features.internal_xr_revision_features import (
    FORBIDDEN_FEATURE_TOKENS,
    METADATA_COLUMNS,
    extract_ecg_features,
    extract_eeg_features,
    extract_pupil_features,
    feature_columns,
    validate_feature_table,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TABLE_ORDER = [
    "EEG",
    "ECG",
    "Pupil",
    "ECG_EEG",
    "EEG_Pupil",
    "ECG_Pupil",
    "ECG_EEG_Pupil",
]

TABLE_FILENAMES = {
    table_name: f"{table_name}_features.csv"
    for table_name in TABLE_ORDER
}


def resolve_project_path(
    value: str | Path,
) -> Path:
    path = Path(value)

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path.resolve()


def parse_boolean(value: Any) -> bool:
    """Parse serialized Boolean values without Python truthiness errors."""

    if isinstance(value, (bool, np.bool_)):
        return bool(value)

    normalized = str(value).strip().lower()

    if normalized in {"true", "1", "yes"}:
        return True

    if normalized in {"false", "0", "no"}:
        return False

    raise ValueError(
        f"Unrecognized Boolean value: {value!r}"
    )


def metadata_record(
    row: Any,
) -> dict[str, Any]:
    """Map external event metadata to the locked internal table schema."""

    return {
        "segment_id": str(row.segment_id),
        "participant": str(row.participant),
        "phase": "memory",
        "label": int(row.label),
        "window_index": int(
            row.memory_trial_index
        ) - 1,
        "pupil_available": parse_boolean(
            row.pupil_available
        ),
    }


def empty_feature_table(
    feature_names: list[str],
) -> pd.DataFrame:
    """Create a schema-valid zero-row feature table."""

    columns = [
        *METADATA_COLUMNS,
        *feature_names,
    ]

    if len(columns) != len(set(columns)):
        raise ValueError(
            "Empty feature-table schema contains duplicate columns."
        )

    return pd.DataFrame(
        columns=columns
    )


def fuse_feature_tables(
    base: pd.DataFrame,
    additional: list[pd.DataFrame],
    table_name: str,
) -> pd.DataFrame:
    """Fuse feature blocks using the base table's ordered segment subset."""

    if base["segment_id"].duplicated().any():
        raise ValueError(
            f"{table_name}: duplicate base segment identifiers."
        )

    result = base[
        METADATA_COLUMNS
    ].copy()

    ordered_ids = result[
        "segment_id"
    ].astype(str).tolist()

    for frame in [base, *additional]:
        if frame["segment_id"].duplicated().any():
            raise ValueError(
                f"{table_name}: duplicate segment identifiers."
            )

        indexed = frame.set_index(
            frame["segment_id"].astype(str),
            drop=False,
        )

        missing = [
            segment_id
            for segment_id in ordered_ids
            if segment_id not in indexed.index
        ]

        if missing:
            raise ValueError(
                f"{table_name}: missing {len(missing)} "
                "segments during fusion."
            )

        aligned = indexed.loc[
            ordered_ids
        ].reset_index(drop=True)

        for metadata_column in [
            "participant",
            "phase",
            "label",
            "window_index",
            "pupil_available",
        ]:
            left = result[
                metadata_column
            ].astype(str).to_numpy()

            right = aligned[
                metadata_column
            ].astype(str).to_numpy()

            if not np.array_equal(left, right):
                raise ValueError(
                    f"{table_name}: metadata mismatch in "
                    f"{metadata_column}."
                )

        block_columns = feature_columns(
            aligned
        )

        overlap = set(result.columns) & set(
            block_columns
        )

        if overlap:
            raise ValueError(
                f"{table_name}: duplicate feature names "
                f"{sorted(overlap)}"
            )

        result = pd.concat(
            [
                result.reset_index(drop=True),
                aligned[
                    block_columns
                ].reset_index(drop=True),
            ],
            axis=1,
        )

    validate_feature_table(
        result,
        table_name,
    )

    return result


def extract_subject_tables(
    subject_directory: str | Path,
    config: dict[str, Any],
) -> dict[str, pd.DataFrame]:
    """Extract all seven modality-path tables for one participant."""

    root = Path(subject_directory)

    metadata = pd.read_csv(
        root / "metadata.csv",
        low_memory=False,
    )

    with np.load(
        root / "windows.npz",
        allow_pickle=False,
    ) as archive:
        eeg = archive["eeg"]
        ecg = archive["ecg"]
        pupil = archive["pupil"]

    if len(metadata) != len(eeg):
        raise RuntimeError(
            f"{root.name}: metadata and EEG lengths differ."
        )

    if len(ecg) != len(metadata):
        raise RuntimeError(
            f"{root.name}: metadata and ECG lengths differ."
        )

    if len(pupil) != len(metadata):
        raise RuntimeError(
            f"{root.name}: metadata and pupil lengths differ."
        )

    channels = list(
        config["eeg"][
            "ds003838_common_channels"
        ]
    )

    if eeg.shape[2] != len(channels):
        raise RuntimeError(
            f"{root.name}: EEG channel count differs from config."
        )

    feature_config = config["features"]

    eeg_bands = feature_config[
        "eeg_bands_hz"
    ]

    pupil_bands = feature_config[
        "pupil_bands_hz"
    ]

    nperseg = int(
        feature_config.get(
            "welch_nperseg",
            256,
        )
    )

    eeg_rate_hz = float(
        config["datasets"]["ds003838"][
            "preprocessing"
        ][
            "eeg_ecg_target_sampling_rate_hz"
        ]
    )

    pupil_rate_hz = float(
        config["datasets"]["ds003838"][
            "preprocessing"
        ][
            "pupil_target_sampling_rate_hz"
        ]
    )

    eeg_rows: list[dict[str, Any]] = []
    ecg_rows: list[dict[str, Any]] = []
    pupil_rows: list[dict[str, Any]] = []

    for window_index, row in enumerate(
        metadata.itertuples(index=False)
    ):
        base = metadata_record(row)

        eeg_features = extract_eeg_features(
            eeg[window_index],
            eeg_rate_hz,
            channels,
            eeg_bands,
            nperseg,
        )

        ecg_features = extract_ecg_features(
            ecg[window_index],
            eeg_rate_hz,
            nperseg,
        )

        eeg_rows.append(
            {
                **base,
                **eeg_features,
            }
        )

        ecg_rows.append(
            {
                **base,
                **ecg_features,
            }
        )

        if base["pupil_available"]:
            if not np.isfinite(
                pupil[window_index]
            ).all():
                raise RuntimeError(
                    f"{root.name}: available pupil "
                    f"window {window_index} is nonfinite."
                )

            pupil_features = extract_pupil_features(
                pupil[window_index],
                pupil_rate_hz,
                pupil_bands,
                nperseg,
            )

            pupil_rows.append(
                {
                    **base,
                    **pupil_features,
                }
            )

    eeg_table = pd.DataFrame(eeg_rows)
    ecg_table = pd.DataFrame(ecg_rows)

    if pupil_rows:
        pupil_table = pd.DataFrame(
            pupil_rows
        )

    else:
        expected_pupil_samples = int(
            round(
                float(
                    config["datasets"]["ds003838"][
                        "window"
                    ][
                        "duration_seconds"
                    ]
                )
                * pupil_rate_hz
            )
        )

        pupil_template = extract_pupil_features(
            np.zeros(
                (
                    expected_pupil_samples,
                    3,
                ),
                dtype=float,
            ),
            pupil_rate_hz,
            pupil_bands,
            nperseg,
        )

        pupil_table = empty_feature_table(
            list(
                pupil_template.keys()
            )
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

    tables = {
        "EEG": eeg_table,
        "ECG": ecg_table,
        "Pupil": pupil_table,
        "ECG_EEG": fuse_feature_tables(
            ecg_table,
            [eeg_table],
            "ECG_EEG",
        ),
        "EEG_Pupil": fuse_feature_tables(
            pupil_table,
            [eeg_table],
            "EEG_Pupil",
        ),
        "ECG_Pupil": fuse_feature_tables(
            pupil_table,
            [ecg_table],
            "ECG_Pupil",
        ),
        "ECG_EEG_Pupil": fuse_feature_tables(
            pupil_table,
            [
                ecg_table,
                eeg_table,
            ],
            "ECG_EEG_Pupil",
        ),
    }

    expected_full_rows = int(len(metadata))

    expected_pupil_rows = int(
        metadata[
            "pupil_available"
        ]
        .map(parse_boolean)
        .sum()
    )

    for table_name in [
        "EEG",
        "ECG",
        "ECG_EEG",
    ]:
        if len(tables[table_name]) != expected_full_rows:
            raise RuntimeError(
                f"{root.name}: {table_name} row count changed."
            )

    for table_name in [
        "Pupil",
        "EEG_Pupil",
        "ECG_Pupil",
        "ECG_EEG_Pupil",
    ]:
        if len(tables[table_name]) != expected_pupil_rows:
            raise RuntimeError(
                f"{root.name}: {table_name} row count changed."
            )

    return tables


def subject_summary(
    subject: str,
    tables: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    """Summarize one verified feature shard."""

    return {
        "subject": subject,
        "tables": {
            table_name: {
                "rows": int(len(table)),
                "features": int(
                    len(
                        feature_columns(table)
                    )
                ),
                "label_counts": {
                    str(key): int(value)
                    for key, value in sorted(
                        table["label"]
                        .astype(int)
                        .value_counts()
                        .to_dict()
                        .items()
                    )
                },
            }
            for table_name, table in tables.items()
        },
    }


def write_subject_shard(
    subject: str,
    tables: dict[str, pd.DataFrame],
    output_directory: Path,
) -> None:
    """Write a participant feature shard atomically."""

    if output_directory.exists():
        raise FileExistsError(
            f"Refusing to overwrite feature shard: "
            f"{output_directory}"
        )

    temporary = output_directory.with_name(
        output_directory.name + ".building"
    )

    if temporary.exists():
        raise FileExistsError(
            f"Temporary shard already exists: {temporary}"
        )

    temporary.mkdir(
        parents=True,
        exist_ok=False,
    )

    for table_name in TABLE_ORDER:
        tables[table_name].to_csv(
            temporary / TABLE_FILENAMES[
                table_name
            ],
            index=False,
        )

    (
        temporary / "summary.json"
    ).write_text(
        json.dumps(
            subject_summary(
                subject,
                tables,
            ),
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )

    temporary.replace(
        output_directory
    )


def verify_subject_shard(
    subject: str,
    shard_directory: Path,
) -> dict[str, Any]:
    """Verify an existing feature shard before skipping it."""

    required = [
        shard_directory / TABLE_FILENAMES[
            table_name
        ]
        for table_name in TABLE_ORDER
    ]

    required.append(
        shard_directory / "summary.json"
    )

    missing = [
        str(path)
        for path in required
        if not path.is_file()
    ]

    if missing:
        raise RuntimeError(
            f"{subject}: incomplete feature shard: {missing}"
        )

    summary = json.loads(
        (
            shard_directory / "summary.json"
        ).read_text(encoding="utf-8")
    )

    if summary["subject"] != subject:
        raise RuntimeError(
            f"{subject}: shard subject mismatch."
        )

    schemas: dict[str, list[str]] = {}

    for table_name in TABLE_ORDER:
        table = pd.read_csv(
            shard_directory
            / TABLE_FILENAMES[table_name],
            low_memory=False,
        )

        validate_feature_table(
            table,
            table_name,
        )

        expected_rows = int(
            summary["tables"][
                table_name
            ][
                "rows"
            ]
        )

        if len(table) != expected_rows:
            raise RuntimeError(
                f"{subject}: {table_name} row count "
                "differs from shard summary."
            )

        schemas[table_name] = feature_columns(
            table
        )

    return {
        "summary": summary,
        "schemas": schemas,
    }


def assert_schema_match(
    expected: dict[str, list[str]],
    observed: dict[str, list[str]],
    subject: str,
) -> None:
    """Require identical ordered feature schemas across participants."""

    for table_name in TABLE_ORDER:
        if (
            expected[table_name]
            != observed[table_name]
        ):
            raise RuntimeError(
                f"{subject}: {table_name} feature schema changed."
            )


def write_aggregate_table(
    frames: list[pd.DataFrame],
    destination: Path,
    table_name: str,
) -> pd.DataFrame:
    """Validate and atomically write one cohort feature table."""

    table = pd.concat(
        frames,
        ignore_index=True,
    )

    validate_feature_table(
        table,
        table_name,
    )

    if table["segment_id"].duplicated().any():
        raise RuntimeError(
            f"{table_name}: duplicate segment identifiers."
        )

    temporary = destination.with_suffix(
        destination.suffix + ".building"
    )

    if temporary.exists():
        raise FileExistsError(
            f"Temporary aggregate exists: {temporary}"
        )

    if destination.exists():
        raise FileExistsError(
            f"Refusing to overwrite aggregate: {destination}"
        )

    table.to_csv(
        temporary,
        index=False,
    )

    temporary.replace(destination)

    return table


def run_feature_cohort(
    config: dict[str, Any],
    *,
    preprocessed_root: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    """Extract resumable feature shards and aggregate the cohort."""

    source = resolve_project_path(
        preprocessed_root
    )

    destination = resolve_project_path(
        output_root
    )

    destination.mkdir(
        parents=True,
        exist_ok=True,
    )

    shard_root = destination / "shards"

    shard_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    cohort_subjects = pd.read_csv(
        source / "cohort_subjects.csv",
        low_memory=False,
    )

    subjects = sorted(
        cohort_subjects[
            "subject"
        ].astype(str).tolist()
    )

    if len(subjects) != 58:
        raise RuntimeError(
            f"Expected 58 participants, observed {len(subjects)}."
        )

    reference_schema: dict[
        str,
        list[str]
    ] | None = None

    total = len(subjects)

    for index, subject in enumerate(subjects):
        source_directory = source / subject
        shard_directory = shard_root / subject

        print(
            f"\n[{index + 1:02d}/{total:02d}] {subject}",
            flush=True,
        )

        if not shard_directory.exists():
            tables = extract_subject_tables(
                source_directory,
                config,
            )

            write_subject_shard(
                subject,
                tables,
                shard_directory,
            )

            print(
                "Feature shard extracted.",
                flush=True,
            )

        else:
            print(
                "Existing shard found; verifying before skip.",
                flush=True,
            )

        verified = verify_subject_shard(
            subject,
            shard_directory,
        )

        if reference_schema is None:
            reference_schema = verified[
                "schemas"
            ]

        else:
            assert_schema_match(
                reference_schema,
                verified["schemas"],
                subject,
            )

        print(
            "Verified: "
            f"ECG_EEG="
            f"{verified['summary']['tables']['ECG_EEG']['rows']}, "
            f"trimodal="
            f"{verified['summary']['tables']['ECG_EEG_Pupil']['rows']}",
            flush=True,
        )

    if reference_schema is None:
        raise RuntimeError(
            "No feature shards were produced."
        )

    aggregate_tables: dict[
        str,
        pd.DataFrame
    ] = {}

    print(
        "\n===== AGGREGATE COHORT FEATURE TABLES =====",
        flush=True,
    )

    for table_name in TABLE_ORDER:
        destination_path = (
            destination
            / TABLE_FILENAMES[table_name]
        )

        if destination_path.exists():
            raise FileExistsError(
                f"Aggregate already exists: {destination_path}"
            )

        frames = [
            pd.read_csv(
                shard_root
                / subject
                / TABLE_FILENAMES[table_name],
                low_memory=False,
            )
            for subject in subjects
        ]

        aggregate_tables[
            table_name
        ] = write_aggregate_table(
            frames,
            destination_path,
            table_name,
        )

        print(
            f"{table_name}: "
            f"rows={len(aggregate_tables[table_name])}, "
            f"features="
            f"{len(feature_columns(aggregate_tables[table_name]))}",
            flush=True,
        )

    full_tables = [
        "EEG",
        "ECG",
        "ECG_EEG",
    ]

    pupil_tables = [
        "Pupil",
        "EEG_Pupil",
        "ECG_Pupil",
        "ECG_EEG_Pupil",
    ]

    for table_name in full_tables:
        table = aggregate_tables[
            table_name
        ]

        if len(table) != 6264:
            raise RuntimeError(
                f"{table_name}: expected 6264 rows."
            )

        if (
            table["participant"].nunique()
            != 58
        ):
            raise RuntimeError(
                f"{table_name}: expected 58 participants."
            )

        if (
            table["label"]
            .astype(int)
            .value_counts()
            .sort_index()
            .to_dict()
            != {
                0: 2088,
                1: 2088,
                2: 2088,
            }
        ):
            raise RuntimeError(
                f"{table_name}: labels are not balanced."
            )

    for table_name in pupil_tables:
        table = aggregate_tables[
            table_name
        ]

        if len(table) != 2368:
            raise RuntimeError(
                f"{table_name}: expected 2368 rows."
            )

        if (
            table["label"]
            .astype(int)
            .value_counts()
            .sort_index()
            .to_dict()
            != {
                0: 814,
                1: 773,
                2: 781,
            }
        ):
            raise RuntimeError(
                f"{table_name}: pupil-subset labels changed."
            )

    all_feature_names = {
        column
        for table in aggregate_tables.values()
        for column in feature_columns(table)
    }

    forbidden_found = sorted(
        feature_name
        for feature_name in all_feature_names
        if any(
            token in feature_name.lower()
            for token in FORBIDDEN_FEATURE_TOKENS
        )
    )

    if forbidden_found:
        raise RuntimeError(
            "Forbidden feature names detected: "
            f"{forbidden_found}"
        )

    summary = {
        "dataset": "ds003838",
        "cohort": "strict_primary",
        "primary_external_modality": "ECG_EEG",
        "primary_participant_count": 58,
        "primary_window_count": 6264,
        "primary_label_counts": {
            "0": 2088,
            "1": 2088,
            "2": 2088,
        },
        "pupil_dependent_analysis_role": (
            "secondary_sensitivity"
        ),
        "pupil_dependent_window_count": 2368,
        "pupil_dependent_participant_count": int(
            aggregate_tables[
                "Pupil"
            ][
                "participant"
            ].nunique()
        ),
        "pupil_dependent_label_counts": {
            "0": 814,
            "1": 773,
            "2": 781,
        },
        "welch_nperseg": int(
            config["features"][
                "welch_nperseg"
            ]
        ),
        "eeg_bands_hz": config[
            "features"
        ][
            "eeg_bands_hz"
        ],
        "pupil_bands_hz": config[
            "features"
        ][
            "pupil_bands_hz"
        ],
        "eeg_channels": list(
            config["eeg"][
                "ds003838_common_channels"
            ]
        ),
        "feature_cleanup_scope": config[
            "features"
        ][
            "feature_cleanup_scope"
        ],
        "excluded_feature_families": [
            "frequency-domain HRV from four-second ECG windows",
            "legacy PDE-named features",
            "high gamma above 40 Hz",
            "absolute gaze-position means",
        ],
        "tables": {
            table_name: {
                "filename": TABLE_FILENAMES[
                    table_name
                ],
                "rows": int(
                    len(table)
                ),
                "participants": int(
                    table[
                        "participant"
                    ].nunique()
                ),
                "features": int(
                    len(
                        feature_columns(table)
                    )
                ),
                "label_counts": {
                    str(key): int(value)
                    for key, value in sorted(
                        table["label"]
                        .astype(int)
                        .value_counts()
                        .to_dict()
                        .items()
                    )
                },
            }
            for table_name, table
            in aggregate_tables.items()
        },
    }

    (
        destination
        / "feature_schemas.json"
    ).write_text(
        json.dumps(
            reference_schema,
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )

    (
        destination
        / "feature_summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )

    print(
        "\n===== DS003838 FEATURE EXTRACTION COMPLETE ====="
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
            "Extract corrected ds003838 strict-primary "
            "feature tables."
        )
    )

    parser.add_argument(
        "--preprocessed-root",
        default=(
            "outputs/revision/preprocessed/"
            "ds003838_strict_primary"
        ),
    )

    parser.add_argument(
        "--output-root",
        default=(
            "outputs/revision/features/"
            "ds003838_strict_primary"
        ),
    )

    arguments = parser.parse_args()

    config = load_revision_config(
        check_input_paths=True
    )

    run_feature_cohort(
        config,
        preprocessed_root=arguments.preprocessed_root,
        output_root=arguments.output_root,
    )


if __name__ == "__main__":
    main()