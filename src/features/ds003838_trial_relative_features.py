"""Build trial-relative ds003838 ECG+EEG feature cohorts."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.core.config import load_revision_config
from src.features.ds003838_revision_features import (
    extract_subject_tables,
)
from src.features.internal_xr_revision_features import (
    feature_columns,
    validate_feature_table,
)
from src.preprocessing.ds003838_cohort import (
    verify_subject_output,
)
from src.preprocessing.ds003838_revision import (
    preprocess_subject,
    write_subject_output,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PRIMARY_TABLE_NAME = "ECG_EEG"
EXPECTED_PARTICIPANTS = 58
EXPECTED_ROWS_PER_PARTICIPANT = 108
EXPECTED_FEATURES = 219
EXPECTED_LABEL_COUNTS = {
    0: 2088,
    1: 2088,
    2: 2088,
}

CORE_ALIGNMENT_COLUMNS = [
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


def parse_boolean_series(
    series: pd.Series,
) -> pd.Series:
    """Parse manifest booleans without Python truthiness."""

    normalized = (
        series.astype(str)
        .str.strip()
        .str.lower()
    )

    accepted = {
        "true": True,
        "1": True,
        "false": False,
        "0": False,
    }

    unknown = sorted(
        set(normalized)
        - set(accepted)
    )

    if unknown:
        raise ValueError(
            f"Unknown boolean values: {unknown}"
        )

    return normalized.map(
        accepted
    ).astype(bool)


def build_baseline_config(
    config: dict[str, Any],
) -> dict[str, Any]:
    """Create the locked pre-first-digit window configuration."""

    baseline = copy.deepcopy(
        config
    )

    window = baseline[
        "datasets"
    ][
        "ds003838"
    ][
        "window"
    ]

    window["anchor"] = (
        "first_digit_onset"
    )

    window[
        "start_offset_seconds"
    ] = -4.5

    window[
        "duration_seconds"
    ] = 4.0

    return baseline


def align_feature_tables(
    post: pd.DataFrame,
    baseline: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    list[str],
]:
    """Align post-task and baseline feature tables exactly."""

    post_aligned = (
        post.sort_values(
            "segment_id"
        )
        .reset_index(drop=True)
    )

    baseline_aligned = (
        baseline.sort_values(
            "segment_id"
        )
        .reset_index(drop=True)
    )

    if len(post_aligned) != len(
        baseline_aligned
    ):
        raise RuntimeError(
            "Post and baseline row counts differ."
        )

    for column in CORE_ALIGNMENT_COLUMNS:
        if column not in post_aligned:
            raise RuntimeError(
                f"Post table lacks metadata column: {column}"
            )

        if column not in baseline_aligned:
            raise RuntimeError(
                f"Baseline table lacks metadata column: {column}"
            )

        left = post_aligned[
            column
        ].astype(str)

        right = baseline_aligned[
            column
        ].astype(str)

        if not left.equals(right):
            raise RuntimeError(
                f"Post and baseline tables differ in {column}."
            )

    post_features = feature_columns(
        post_aligned
    )

    baseline_features = feature_columns(
        baseline_aligned
    )

    if post_features != baseline_features:
        raise RuntimeError(
            "Post and baseline feature schemas differ."
        )

    if not post_features:
        raise RuntimeError(
            "No aligned features were found."
        )

    return (
        post_aligned,
        baseline_aligned,
        post_features,
    )


def build_delta_table(
    post: pd.DataFrame,
    baseline: pd.DataFrame,
) -> pd.DataFrame:
    """Construct post-minus-pre features without frame fragmentation."""

    (
        post_aligned,
        baseline_aligned,
        source_features,
    ) = align_feature_tables(
        post,
        baseline,
    )

    post_values = post_aligned[
        source_features
    ].to_numpy(dtype=float)

    baseline_values = baseline_aligned[
        source_features
    ].to_numpy(dtype=float)

    if not np.isfinite(
        post_values
    ).all():
        raise RuntimeError(
            "Post-task features contain nonfinite values."
        )

    if not np.isfinite(
        baseline_values
    ).all():
        raise RuntimeError(
            "Baseline features contain nonfinite values."
        )

    delta_values = (
        post_values
        - baseline_values
    )

    delta_columns = [
        f"delta_{feature}"
        for feature in source_features
    ]

    metadata_columns = [
        column
        for column in post_aligned.columns
        if column not in source_features
    ]

    metadata = (
        post_aligned[
            metadata_columns
        ]
        .copy()
        .reset_index(drop=True)
    )

    delta_features = pd.DataFrame(
        delta_values,
        columns=delta_columns,
    )

    result = pd.concat(
        [
            metadata,
            delta_features,
        ],
        axis=1,
    )

    validate_feature_table(
        result,
        "ds003838_trial_relative_ECG_EEG",
    )

    return result


def verify_delta_shard(
    shard_directory: Path,
    *,
    expected_subject: str,
) -> dict[str, Any]:
    """Verify one completed trial-relative participant shard."""

    table_path = (
        shard_directory
        / "ECG_EEG_delta_features.csv"
    )

    summary_path = (
        shard_directory
        / "summary.json"
    )

    required = [
        table_path,
        summary_path,
    ]

    missing = [
        str(path)
        for path in required
        if not path.is_file()
    ]

    if missing:
        raise RuntimeError(
            f"{expected_subject}: incomplete delta shard: {missing}"
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
            f"{expected_subject}: expected "
            f"{EXPECTED_ROWS_PER_PARTICIPANT} rows, "
            f"observed {len(table)}."
        )

    if len(features) != EXPECTED_FEATURES:
        raise RuntimeError(
            f"{expected_subject}: expected "
            f"{EXPECTED_FEATURES} features, "
            f"observed {len(features)}."
        )

    if set(
        table["participant"].astype(str)
    ) != {
        expected_subject
    }:
        raise RuntimeError(
            f"{expected_subject}: participant registry changed."
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
            f"{expected_subject}: labels are not balanced."
        )

    if table[
        "segment_id"
    ].duplicated().any():
        raise RuntimeError(
            f"{expected_subject}: duplicate segment identifiers."
        )

    values = table[
        features
    ].to_numpy(dtype=float)

    if not np.isfinite(
        values
    ).all():
        raise RuntimeError(
            f"{expected_subject}: nonfinite delta values detected."
        )

    if summary[
        "representation"
    ] != "trial_relative_post_minus_pre":
        raise RuntimeError(
            f"{expected_subject}: representation contract changed."
        )

    return {
        "subject": expected_subject,
        "rows": int(len(table)),
        "features": int(len(features)),
        "median_absolute_delta": float(
            np.median(
                np.abs(values)
            )
        ),
    }


def write_delta_shard(
    subject: str,
    table: pd.DataFrame,
    destination: Path,
) -> None:
    """Write one participant shard atomically."""

    if destination.exists():
        raise FileExistsError(
            f"Refusing to overwrite delta shard: {destination}"
        )

    temporary = destination.with_name(
        destination.name + ".building"
    )

    if temporary.exists():
        raise FileExistsError(
            f"Temporary delta shard already exists: {temporary}"
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

    table.to_csv(
        temporary
        / "ECG_EEG_delta_features.csv",
        index=False,
    )

    summary = {
        "dataset": "ds003838",
        "subject": subject,
        "representation":
            "trial_relative_post_minus_pre",
        "post_window": {
            "anchor":
                "final_digit_offset",
            "start_offset_seconds":
                1.0,
            "duration_seconds":
                4.0,
        },
        "baseline_window": {
            "anchor":
                "first_digit_onset",
            "start_offset_seconds":
                -4.5,
            "duration_seconds":
                4.0,
            "end_margin_seconds":
                0.5,
        },
        "rows": int(len(table)),
        "feature_count":
            int(len(features)),
        "label_counts": {
            str(key): int(value)
            for key, value
            in table[
                "label"
            ]
            .astype(int)
            .value_counts()
            .sort_index()
            .to_dict()
            .items()
        },
        "median_absolute_delta":
            float(
                np.median(
                    np.abs(values)
                )
            ),
        "all_values_finite":
            bool(
                np.isfinite(
                    values
                ).all()
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


def eta_squared_effects(
    table: pd.DataFrame,
) -> pd.DataFrame:
    """Compare participant and class effects feature by feature."""

    features = feature_columns(
        table
    )

    participant_codes = (
        table["participant"]
        .astype("category")
        .cat.codes
        .to_numpy(dtype=int)
    )

    labels = table[
        "label"
    ].to_numpy(dtype=int)

    rows = []

    for feature in features:
        values = table[
            feature
        ].to_numpy(dtype=float)

        grand_mean = float(
            np.mean(values)
        )

        total_ss = float(
            np.sum(
                (
                    values
                    - grand_mean
                ) ** 2
            )
        )

        if total_ss <= 1e-20:
            participant_eta = 0.0
            class_eta = 0.0

        else:
            participant_ss = 0.0

            for participant_code in np.unique(
                participant_codes
            ):
                mask = (
                    participant_codes
                    == participant_code
                )

                participant_ss += (
                    int(mask.sum())
                    * (
                        float(
                            np.mean(
                                values[mask]
                            )
                        )
                        - grand_mean
                    ) ** 2
                )

            class_ss = 0.0

            for label in [0, 1, 2]:
                mask = labels == label

                class_ss += (
                    int(mask.sum())
                    * (
                        float(
                            np.mean(
                                values[mask]
                            )
                        )
                        - grand_mean
                    ) ** 2
                )

            participant_eta = (
                participant_ss
                / total_ss
            )

            class_eta = (
                class_ss
                / total_ss
            )

        rows.append(
            {
                "feature": feature,
                "participant_eta_squared":
                    float(participant_eta),
                "class_eta_squared":
                    float(class_eta),
                "participant_to_class_ratio":
                    float(
                        participant_eta
                        / (
                            class_eta
                            + 1e-20
                        )
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


def run_trial_relative_cohort(
    config: dict[str, Any],
    *,
    post_root: str | Path,
    baseline_root: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    """Build or resume the complete trial-relative primary cohort."""

    post_directory = resolve_project_path(
        post_root
    )

    baseline_directory = resolve_project_path(
        baseline_root
    )

    destination = resolve_project_path(
        output_root
    )

    destination.mkdir(
        parents=True,
        exist_ok=True,
    )

    baseline_directory.mkdir(
        parents=True,
        exist_ok=True,
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
        .sort_values(
            "subject"
        )
        .reset_index(drop=True)
    )

    if len(strict) != EXPECTED_PARTICIPANTS:
        raise RuntimeError(
            f"Expected {EXPECTED_PARTICIPANTS} participants, "
            f"observed {len(strict)}."
        )

    baseline_config = build_baseline_config(
        config
    )

    shard_root = (
        destination / "subjects"
    )

    shard_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    shard_summaries = []

    for index, manifest_row in strict.iterrows():
        subject = str(
            manifest_row["subject"]
        )

        print(
            f"\n[{index + 1:02d}/{EXPECTED_PARTICIPANTS:02d}] "
            f"{subject}",
            flush=True,
        )

        baseline_subject = (
            baseline_directory
            / subject
        )

        if not baseline_subject.exists():
            print(
                "  Preprocessing pretrial baseline windows.",
                flush=True,
            )

            arrays, metadata, audit = (
                preprocess_subject(
                    manifest_row,
                    baseline_config,
                )
            )

            write_subject_output(
                arrays,
                metadata,
                audit,
                baseline_subject,
            )

        else:
            print(
                "  Existing baseline windows found; verifying.",
                flush=True,
            )

        verify_subject_output(
            baseline_subject,
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
                "  Extracting post and baseline features.",
                flush=True,
            )

            post_tables = extract_subject_tables(
                post_directory / subject,
                config,
            )

            baseline_tables = (
                extract_subject_tables(
                    baseline_subject,
                    baseline_config,
                )
            )

            delta = build_delta_table(
                post_tables[
                    PRIMARY_TABLE_NAME
                ],
                baseline_tables[
                    PRIMARY_TABLE_NAME
                ],
            )

            write_delta_shard(
                subject,
                delta,
                shard_directory,
            )

        else:
            print(
                "  Existing delta shard found; verifying.",
                flush=True,
            )

        verified = verify_delta_shard(
            shard_directory,
            expected_subject=subject,
        )

        shard_summaries.append(
            verified
        )

        print(
            "  Verified: "
            f"{verified['rows']} rows, "
            f"{verified['features']} features, "
            f"median |delta|="
            f"{verified['median_absolute_delta']:.8f}",
            flush=True,
        )

    tables = []

    for subject in strict[
        "subject"
    ].astype(str):
        table = pd.read_csv(
            shard_root
            / subject
            / "ECG_EEG_delta_features.csv",
            low_memory=False,
        )

        tables.append(
            table
        )

    aggregate = pd.concat(
        tables,
        ignore_index=True,
    )

    aggregate_features = feature_columns(
        aggregate
    )

    if len(aggregate) != 6264:
        raise RuntimeError(
            f"Expected 6264 aggregate rows, "
            f"observed {len(aggregate)}."
        )

    if aggregate[
        "participant"
    ].nunique() != EXPECTED_PARTICIPANTS:
        raise RuntimeError(
            "Aggregate participant count changed."
        )

    if len(
        aggregate_features
    ) != EXPECTED_FEATURES:
        raise RuntimeError(
            f"Expected {EXPECTED_FEATURES} aggregate features, "
            f"observed {len(aggregate_features)}."
        )

    if (
        aggregate["label"]
        .astype(int)
        .value_counts()
        .sort_index()
        .to_dict()
        != EXPECTED_LABEL_COUNTS
    ):
        raise RuntimeError(
            "Aggregate label counts changed."
        )

    if aggregate[
        "segment_id"
    ].duplicated().any():
        raise RuntimeError(
            "Aggregate segment identifiers are duplicated."
        )

    aggregate_values = aggregate[
        aggregate_features
    ].to_numpy(dtype=float)

    if not np.isfinite(
        aggregate_values
    ).all():
        raise RuntimeError(
            "Aggregate delta values contain nonfinite entries."
        )

    effects = eta_squared_effects(
        aggregate
    )

    absolute_table = pd.read_csv(
        resolve_project_path(
            "outputs/revision/features/"
            "ds003838_strict_primary/"
            "ECG_EEG_features.csv"
        ),
        low_memory=False,
    )

    absolute_effects = eta_squared_effects(
        absolute_table
    )

    delta_effect_summary = {
        "median_participant_eta_squared":
            float(
                effects[
                    "participant_eta_squared"
                ].median()
            ),
        "median_class_eta_squared":
            float(
                effects[
                    "class_eta_squared"
                ].median()
            ),
        "features_participant_effect_greater_than_class_effect":
            int(
                (
                    effects[
                        "participant_eta_squared"
                    ]
                    > effects[
                        "class_eta_squared"
                    ]
                ).sum()
            ),
        "median_participant_to_class_ratio":
            float(
                effects[
                    "participant_to_class_ratio"
                ].median()
            ),
    }

    absolute_effect_summary = {
        "median_participant_eta_squared":
            float(
                absolute_effects[
                    "participant_eta_squared"
                ].median()
            ),
        "median_class_eta_squared":
            float(
                absolute_effects[
                    "class_eta_squared"
                ].median()
            ),
        "features_participant_effect_greater_than_class_effect":
            int(
                (
                    absolute_effects[
                        "participant_eta_squared"
                    ]
                    > absolute_effects[
                        "class_eta_squared"
                    ]
                ).sum()
            ),
        "median_participant_to_class_ratio":
            float(
                absolute_effects[
                    "participant_to_class_ratio"
                ].median()
            ),
    }

    summary = {
        "dataset": "ds003838",
        "cohort": "strict_primary",
        "representation":
            "trial_relative_post_minus_pre",
        "participant_count":
            EXPECTED_PARTICIPANTS,
        "row_count":
            int(len(aggregate)),
        "feature_count":
            int(len(aggregate_features)),
        "label_counts": {
            str(key): int(value)
            for key, value
            in EXPECTED_LABEL_COUNTS.items()
        },
        "post_window": {
            "anchor":
                "final_digit_offset",
            "start_offset_seconds":
                1.0,
            "duration_seconds":
                4.0,
        },
        "baseline_window": {
            "anchor":
                "first_digit_onset",
            "start_offset_seconds":
                -4.5,
            "duration_seconds":
                4.0,
            "end_margin_seconds":
                0.5,
        },
        "delta_definition":
            "post_feature_minus_pretrial_feature",
        "all_values_finite": True,
        "median_absolute_delta":
            float(
                np.median(
                    np.abs(
                        aggregate_values
                    )
                )
            ),
        "absolute_feature_effects":
            absolute_effect_summary,
        "trial_relative_feature_effects":
            delta_effect_summary,
    }

    aggregate_path = (
        destination
        / "ECG_EEG_delta_features.csv"
    )

    effects_path = (
        destination
        / "participant_vs_class_effects.csv"
    )

    subjects_path = (
        destination
        / "cohort_subjects.csv"
    )

    summary_path = (
        destination
        / "cohort_summary.json"
    )

    aggregate.to_csv(
        aggregate_path,
        index=False,
    )

    effects.to_csv(
        effects_path,
        index=False,
    )

    pd.DataFrame(
        shard_summaries
    ).to_csv(
        subjects_path,
        index=False,
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )

    print(
        "\n===== COMPLETE TRIAL-RELATIVE COHORT ====="
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
            "Build the complete ds003838 "
            "trial-relative ECG+EEG cohort."
        )
    )

    parser.add_argument(
        "--post-root",
        default=(
            "outputs/revision/preprocessed/"
            "ds003838_strict_primary"
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
            "ds003838_trial_relative"
        ),
    )

    arguments = parser.parse_args()

    config = load_revision_config(
        check_input_paths=True
    )

    run_trial_relative_cohort(
        config,
        post_root=arguments.post_root,
        baseline_root=
            arguments.baseline_root,
        output_root=
            arguments.output_root,
    )


if __name__ == "__main__":
    main()