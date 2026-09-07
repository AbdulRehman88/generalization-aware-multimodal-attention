
"""Exact real-data bindings for the frozen deep-learning execution plan.

This module performs data selection only.

It deliberately performs:
- no model initialization;
- no optimizer construction;
- no normalization fitting;
- no loss computation;
- no predictions;
- no neural-network fitting;
- no performance evaluation.

The conventional internal index arrays are recovered by executing the
already-frozen execution-plan generator in an isolated temporary
directory while intercepting the exact hash_indices() calls. This
avoids reimplementing or approximating the frozen split logic.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import subprocess
import tempfile
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from src.models.deep_learning_baselines import (
    LOCKED_MODALITY_PATHS,
)
from src.models.deep_learning_training_runtime import (
    expand_execution_registry,
)


ROOT = Path(__file__).resolve().parents[2]

PROTOCOL = (
    ROOT
    / "configs/deep_learning_baselines.yaml"
)

REGISTRY = (
    ROOT
    / "artifacts/revision/manifests/"
      "deep_learning_execution_registry_v2.csv"
)


ACTIVE_REGISTRY_V3 = (
    ROOT
    / "artifacts/revision/manifests/"
      "deep_learning_execution_registry_v3.csv"
)

INTERNAL_MANIFEST = (
    ROOT
    / "outputs/revision/preprocessed/internal_xr/"
      "window_manifest.csv"
)

INTERNAL_CACHE_ROOT = (
    ROOT
    / "outputs/revision/deep_learning_cache_v1/internal_xr"
)

BBBD_CACHE_ROOT = (
    ROOT
    / "outputs/revision/deep_learning_cache_v1/bbbd"
)


INTERNAL_DURATIONS = (
    4,
    8,
    16,
    24,
    32,
)

ADAPTIVE_DURATIONS = (
    8,
    16,
    24,
    32,
)

ADAPTIVE_BUDGETS = (
    30,
    60,
    120,
)


def sha256_indices(
    values,
) -> str:

    array = np.asarray(
        values,
        dtype=np.int64,
    )

    array = np.sort(
        array
    )

    return hashlib.sha256(
        array.tobytes()
    ).hexdigest()


def split_semicolon(
    value,
) -> list[str]:

    if value is None:
        return []

    if isinstance(
        value,
        float,
    ) and np.isnan(
        value
    ):
        return []

    return [
        token.strip()
        for token
        in str(
            value
        ).split(";")
        if token.strip()
    ]


def path_modalities(
    path: str,
) -> tuple[str, ...]:

    if path not in LOCKED_MODALITY_PATHS:
        raise ValueError(
            f"Unknown path: {path}"
        )

    return tuple(
        LOCKED_MODALITY_PATHS[
            path
        ]
    )


def path_requires_pupil(
    path: str,
) -> bool:

    return (
        "Pupil"
        in path_modalities(
            path
        )
    )


def internal_cache_path(
    duration_seconds: int,
) -> Path:

    duration_seconds = int(
        duration_seconds
    )

    if duration_seconds not in INTERNAL_DURATIONS:
        raise ValueError(
            f"Unsupported internal duration: {duration_seconds}"
        )

    return (
        INTERNAL_CACHE_ROOT
        / f"window_{duration_seconds:02d}s.npz"
    )


def load_internal_cache(
    duration_seconds: int,
) -> dict[str, np.ndarray]:

    path = internal_cache_path(
        duration_seconds
    )

    if not path.is_file():
        raise FileNotFoundError(
            path
        )

    with np.load(
        path,
        allow_pickle=False,
    ) as archive:

        payload = {
            key:
                np.asarray(
                    archive[
                        key
                    ]
                )
            for key
            in archive.files
        }

    return payload


def internal_eligibility_mask(
    cache: Mapping[
        str,
        np.ndarray,
    ],
    *,
    path: str,
) -> np.ndarray:

    n = len(
        cache[
            "label"
        ]
    )

    mask = np.ones(
        n,
        dtype=bool,
    )

    if path_requires_pupil(
        path
    ):

        if "pupil_valid" not in cache:
            raise RuntimeError(
                "Internal cache lacks pupil_valid."
            )

        mask &= np.asarray(
            cache[
                "pupil_valid"
            ],
            dtype=bool,
        )

    return mask


def participant_mask(
    values: np.ndarray,
    participants: Sequence[str],
) -> np.ndarray:

    participants = [
        str(
            value
        )
        for value
        in participants
    ]

    return np.isin(
        values.astype(
            str
        ),
        participants,
    )


def assert_index_partition(
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    test_indices: np.ndarray,
    *,
    eligible_indices: np.ndarray | None = None,
) -> None:

    train_set = set(
        np.asarray(
            train_indices,
            dtype=np.int64,
        ).tolist()
    )

    validation_set = set(
        np.asarray(
            validation_indices,
            dtype=np.int64,
        ).tolist()
    )

    test_set = set(
        np.asarray(
            test_indices,
            dtype=np.int64,
        ).tolist()
    )

    if train_set & validation_set:
        raise RuntimeError(
            "Train/validation index overlap."
        )

    if train_set & test_set:
        raise RuntimeError(
            "Train/test index overlap."
        )

    if validation_set & test_set:
        raise RuntimeError(
            "Validation/test index overlap."
        )

    if eligible_indices is not None:

        eligible_set = set(
            np.asarray(
                eligible_indices,
                dtype=np.int64,
            ).tolist()
        )

        union = (
            train_set
            | validation_set
            | test_set
        )

        if union != eligible_set:
            raise RuntimeError(
                "Train/validation/test union does not equal "
                "the eligible window universe."
            )


def class_counts(
    labels: np.ndarray,
    indices: np.ndarray,
) -> dict[int, int]:

    selected = np.asarray(
        labels,
        dtype=np.int64,
    )[
        np.asarray(
            indices,
            dtype=np.int64,
        )
    ]

    values, counts = np.unique(
        selected,
        return_counts=True,
    )

    return {
        int(
            value
        ):
            int(
                count
            )
        for value, count
        in zip(
            values,
            counts,
        )
    }


def assert_class_coverage(
    labels: np.ndarray,
    indices: np.ndarray,
    *,
    expected_classes: Sequence[int],
    context: str,
) -> None:

    observed = set(
        class_counts(
            labels,
            indices,
        )
    )

    expected = set(
        int(
            value
        )
        for value
        in expected_classes
    )

    if observed != expected:
        raise RuntimeError(
            f"{context}: expected classes "
            f"{sorted(expected)}, observed {sorted(observed)}."
        )


def validate_internal_manifest_cache_alignment() -> dict:

    manifest = pd.read_csv(
        INTERNAL_MANIFEST,
        low_memory=False,
    )

    cache = load_internal_cache(
        4
    )

    n = len(
        cache[
            "label"
        ]
    )

    if len(
        manifest
    ) != n:
        raise RuntimeError(
            "Internal manifest/cache row-count mismatch."
        )

    if not np.array_equal(
        manifest[
            "participant"
        ].astype(str).to_numpy(),
        cache[
            "participant"
        ].astype(str),
    ):
        raise RuntimeError(
            "Internal participant positional alignment failed."
        )

    if not np.array_equal(
        manifest[
            "phase"
        ].to_numpy(
            dtype=np.int64
        ),
        cache[
            "phase"
        ].astype(
            np.int64
        ),
    ):
        raise RuntimeError(
            "Internal phase positional alignment failed."
        )

    if not np.array_equal(
        manifest[
            "label"
        ].to_numpy(
            dtype=np.int64
        ),
        cache[
            "label"
        ].astype(
            np.int64
        ),
    ):
        raise RuntimeError(
            "Internal label positional alignment failed."
        )

    if not np.array_equal(
        manifest[
            "archive_window_index"
        ].to_numpy(
            dtype=np.int64
        ),
        cache[
            "source_window_index"
        ].astype(
            np.int64
        ),
    ):
        raise RuntimeError(
            "Internal archive-window positional alignment failed."
        )

    manifest_archives = np.asarray(
        [
            str(
                value
            )
            .replace(
                "\\",
                "/",
            )
            .split(
                "/"
            )[
                -1
            ]
            for value
            in manifest[
                "recording_archive"
            ].astype(str)
        ]
    )

    if not np.array_equal(
        manifest_archives,
        cache[
            "source_archive"
        ].astype(str),
    ):
        raise RuntimeError(
            "Internal source-archive positional alignment failed."
        )

    manifest_pupil = (
        manifest[
            "pupil_available"
        ]
        .astype(
            bool
        )
        .to_numpy()
    )

    if not np.array_equal(
        manifest_pupil,
        cache[
            "pupil_valid"
        ].astype(
            bool
        ),
    ):
        raise RuntimeError(
            "Internal pupil-valid positional alignment failed."
        )

    return {
        "rows":
            n,

        "participants":
            int(
                manifest[
                    "participant"
                ].nunique()
            ),

        "pupil_valid":
            int(
                manifest_pupil.sum()
            ),

        "pupil_invalid":
            int(
                (
                    ~manifest_pupil
                ).sum()
            ),

        "passed":
            True,
    }


def regenerate_frozen_conventional_index_map() -> tuple[
    dict[str, np.ndarray],
    pd.DataFrame,
]:
    """Recover exact index arrays using the historical frozen generator.

    The execution-plan Python source, protocol YAML, and registry v2 are
    taken directly from commit df25c9f, the protocol-1.2 freeze that
    generated the v2 execution registry.

    The historical Python source is executed as an isolated module whose
    __file__ is placed directly inside src/evaluation so its original
    project-root calculation remains unchanged.

    No current execution-plan implementation is used for reconstruction.
    """

    import io
    import subprocess
    import tempfile
    import types


    historical_commit = (
        "df25c9f"
    )


    # ------------------------------------------------------------
    # Load the exact historical files from Git.
    # ------------------------------------------------------------

    historical_plan_bytes = subprocess.check_output(
        [
            "git",
            "show",
            (
                f"{historical_commit}:"
                "src/evaluation/deep_learning_execution_plan.py"
            ),
        ],
        cwd=ROOT,
    )


    historical_protocol_bytes = subprocess.check_output(
        [
            "git",
            "show",
            (
                f"{historical_commit}:"
                "configs/deep_learning_baselines.yaml"
            ),
        ],
        cwd=ROOT,
    )


    historical_registry_bytes = subprocess.check_output(
        [
            "git",
            "show",
            (
                f"{historical_commit}:"
                "artifacts/revision/manifests/"
                "deep_learning_execution_registry_v2.csv"
            ),
        ],
        cwd=ROOT,
    )


    historical_protocol = yaml.safe_load(
        historical_protocol_bytes.decode(
            "utf-8"
        )
    )


    if historical_protocol[
        "protocol_revision"
    ] != "1.2":

        raise RuntimeError(
            "Historical df25c9f protocol is not revision 1.2."
        )


    historical_registry = pd.read_csv(
        io.BytesIO(
            historical_registry_bytes
        ),
        low_memory=False,
    )


    current_registry = pd.read_csv(
        REGISTRY,
        low_memory=False,
    )


    if (
        historical_registry.columns.tolist()
        != current_registry.columns.tolist()
    ):
        raise RuntimeError(
            "Current registry columns differ from df25c9f frozen registry."
        )


    historical_compare = (
        historical_registry
        .fillna(
            ""
        )
        .astype(str)
    )

    current_compare = (
        current_registry
        .fillna(
            ""
        )
        .astype(str)
    )


    if not historical_compare.equals(
        current_compare
    ):

        differences = np.flatnonzero(
            ~(
                historical_compare
                == current_compare
            ).all(
                axis=1
            )
        )

        raise RuntimeError(
            "Current execution registry v2 differs from the exact "
            "df25c9f frozen registry. First differing rows: "
            f"{differences[:20].tolist()}"
        )


    # ------------------------------------------------------------
    # Create a temporary historical module file DIRECTLY beneath
    # src/evaluation. This preserves the original
    # Path(__file__).resolve().parents[2] project-root semantics.
    # ------------------------------------------------------------

    evaluation_directory = (
        ROOT
        / "src/evaluation"
    )


    temporary_plan = None


    captured: dict[
        str,
        np.ndarray,
    ] = {}


    try:

        with tempfile.NamedTemporaryFile(
            mode="wb",
            suffix=".py",
            prefix="_historical_deep_plan_",
            dir=evaluation_directory,
            delete=False,
        ) as stream:

            stream.write(
                historical_plan_bytes
            )

            temporary_plan = Path(
                stream.name
            )


        module = types.ModuleType(
            "src.evaluation._historical_deep_plan_replay"
        )

        module.__file__ = str(
            temporary_plan
        )

        module.__package__ = (
            "src.evaluation"
        )


        code = compile(
            historical_plan_bytes.decode(
                "utf-8"
            ),
            str(
                temporary_plan
            ),
            "exec",
        )


        exec(
            code,
            module.__dict__,
        )


        original_hash_indices = (
            module.hash_indices
        )


        def capture_hash_indices(
            values,
        ) -> str:

            digest = original_hash_indices(
                values
            )

            array = np.sort(
                np.asarray(
                    values,
                    dtype=np.int64,
                )
            )


            if digest in captured:

                if not np.array_equal(
                    captured[
                        digest
                    ],
                    array,
                ):

                    raise RuntimeError(
                        "Historical hash collision mapped to "
                        "different index arrays."
                    )

            else:

                captured[
                    digest
                ] = array


            return digest


        module.hash_indices = (
            capture_hash_indices
        )


        # --------------------------------------------------------
        # All temporary output/protocol files remain beneath ROOT
        # so the historical provenance code's relative_to(ROOT)
        # invariant remains valid.
        # --------------------------------------------------------

        with tempfile.TemporaryDirectory(
            dir=ROOT,
            prefix=".historical_plan_outputs_",
        ) as temp_directory:

            temp_directory = Path(
                temp_directory
            )


            temp_protocol = (
                temp_directory
                / "deep_learning_baselines.yaml"
            )

            temp_protocol.write_bytes(
                historical_protocol_bytes
            )


            temp_registry = (
                temp_directory
                / "deep_learning_execution_registry_v2.csv"
            )

            temp_audit_json = (
                temp_directory
                / "deep_learning_execution_plan_v2.json"
            )

            temp_audit_md = (
                temp_directory
                / "deep_learning_execution_plan_v2.md"
            )


            module.PROTOCOL_PATH = (
                temp_protocol
            )

            module.REGISTRY_PATH = (
                temp_registry
            )

            module.AUDIT_JSON = (
                temp_audit_json
            )

            module.AUDIT_MD = (
                temp_audit_md
            )


            module.main()


            if not temp_registry.is_file():
                raise RuntimeError(
                    "Historical execution-plan replay did not create "
                    "its temporary registry."
                )


            replay = pd.read_csv(
                temp_registry,
                low_memory=False,
            )


    finally:

        if (
            temporary_plan is not None
            and temporary_plan.exists()
        ):

            temporary_plan.unlink()


    # ------------------------------------------------------------
    # Historical replay must reproduce frozen registry exactly.
    # ------------------------------------------------------------

    if replay.columns.tolist() != current_registry.columns.tolist():

        raise RuntimeError(
            "Historical replay/current registry columns differ."
        )


    replay_compare = (
        replay
        .fillna(
            ""
        )
        .astype(str)
    )


    if not replay_compare.equals(
        current_compare
    ):

        differences = np.flatnonzero(
            ~(
                replay_compare
                == current_compare
            ).all(
                axis=1
            )
        )

        raise RuntimeError(
            "Exact historical execution-plan replay did not reproduce "
            "frozen registry v2. First differing rows: "
            f"{differences[:20].tolist()}"
        )


    # ------------------------------------------------------------
    # Recover every index array referenced by conventional jobs.
    # ------------------------------------------------------------

    conventional = current_registry.loc[
        current_registry[
            "stage"
        ].astype(str)
        == "internal_conventional"
    ]


    required_hashes = set()


    for column in [
        "train_index_hash",
        "validation_index_hash",
        "test_index_hash",
    ]:

        required_hashes.update(
            value
            for value
            in conventional[
                column
            ].dropna().astype(str)
            if value
        )


    missing = (
        required_hashes
        - set(
            captured
        )
    )


    if missing:

        raise RuntimeError(
            "Historical replay failed to recover "
            f"{len(missing)} conventional index hashes."
        )


    index_map = {
        digest:
            captured[
                digest
            ]
        for digest
        in required_hashes
    }


    # Final independent hash verification.
    for digest, indices in index_map.items():

        if sha256_indices(
            indices
        ) != digest:

            raise RuntimeError(
                "Recovered historical index array failed its "
                f"SHA-256 verification: {digest}"
            )


    return (
        index_map,
        replay,
    )


def bind_internal_conventional_row(
    row: Mapping,
    index_map: Mapping[
        str,
        np.ndarray,
    ],
) -> dict:

    cache = load_internal_cache(
        4
    )

    labels = cache[
        "label"
    ]

    path = str(
        row[
            "path"
        ]
    )

    eligible_mask = internal_eligibility_mask(
        cache,
        path=path,
    )

    eligible_indices = np.flatnonzero(
        eligible_mask
    )


    def recover(
        column: str,
    ) -> np.ndarray:

        digest = str(
            row[
                column
            ]
        )

        if digest not in index_map:
            raise KeyError(
                f"Missing frozen index hash: {digest}"
            )

        values = np.asarray(
            index_map[
                digest
            ],
            dtype=np.int64,
        )

        if sha256_indices(
            values
        ) != digest:
            raise RuntimeError(
                f"Recovered index array does not match {column}."
            )

        return values


    train_indices = recover(
        "train_index_hash"
    )

    validation_indices = recover(
        "validation_index_hash"
    )

    test_indices = recover(
        "test_index_hash"
    )


    assert_index_partition(
        train_indices,
        validation_indices,
        test_indices,
        eligible_indices=eligible_indices,
    )


    for name, values in [
        (
            "train",
            train_indices,
        ),
        (
            "validation",
            validation_indices,
        ),
        (
            "test",
            test_indices,
        ),
    ]:

        if len(
            values
        ) == 0:
            raise RuntimeError(
                f"Internal conventional {name} partition is empty."
            )

        if not eligible_mask[
            values
        ].all():
            raise RuntimeError(
                f"Internal conventional {name} contains "
                "path-ineligible windows."
            )

        assert_class_coverage(
            labels,
            values,
            expected_classes=(
                0,
                1,
                2,
            ),
            context=(
                f"internal conventional "
                f"{row['protocol']} {row['split_id']} "
                f"{path} {name}"
            ),
        )


    return {
        "train_indices":
            train_indices,

        "validation_indices":
            validation_indices,

        "test_indices":
            test_indices,
    }


def bind_internal_participant_split(
    row: Mapping,
    *,
    duration_seconds: int,
) -> dict:

    cache = load_internal_cache(
        duration_seconds
    )

    path = str(
        row[
            "path"
        ]
    )

    eligible = internal_eligibility_mask(
        cache,
        path=path,
    )

    participants = cache[
        "participant"
    ].astype(str)

    labels = cache[
        "label"
    ]


    train_participants = split_semicolon(
        row[
            "train_participants"
        ]
    )

    validation_participants = split_semicolon(
        row[
            "validation_participants"
        ]
    )

    test_participants = split_semicolon(
        row[
            "test_participants"
        ]
    )


    participant_sets = [
        set(
            train_participants
        ),
        set(
            validation_participants
        ),
        set(
            test_participants
        ),
    ]


    if participant_sets[
        0
    ] & participant_sets[
        1
    ]:
        raise RuntimeError(
            "Participant train/validation overlap."
        )

    if participant_sets[
        0
    ] & participant_sets[
        2
    ]:
        raise RuntimeError(
            "Participant train/test overlap."
        )

    if participant_sets[
        1
    ] & participant_sets[
        2
    ]:
        raise RuntimeError(
            "Participant validation/test overlap."
        )


    train_indices = np.flatnonzero(
        eligible
        & participant_mask(
            participants,
            train_participants,
        )
    )

    validation_indices = np.flatnonzero(
        eligible
        & participant_mask(
            participants,
            validation_participants,
        )
    )

    test_indices = np.flatnonzero(
        eligible
        & participant_mask(
            participants,
            test_participants,
        )
    )


    assert_index_partition(
        train_indices,
        validation_indices,
        test_indices,
    )


    for name, values in [
        (
            "train",
            train_indices,
        ),
        (
            "validation",
            validation_indices,
        ),
        (
            "test",
            test_indices,
        ),
    ]:

        if len(
            values
        ) == 0:
            raise RuntimeError(
                f"{row.get('stage', 'internal_nested_loso')} "
                f"{row['split_id']} "
                f"{path} {duration_seconds}s: "
                f"{name} partition is empty."
            )

        assert_class_coverage(
            labels,
            values,
            expected_classes=(
                0,
                1,
                2,
            ),
            context=(
                f"{row['stage']} {row['split_id']} "
                f"{path} {duration_seconds}s {name}"
            ),
        )


    return {
        "train_indices":
            train_indices,

        "validation_indices":
            validation_indices,

        "test_indices":
            test_indices,
    }


def adaptive_temporal_roles(
    *,
    row: Mapping,
    duration_seconds: int,
) -> dict:

    duration_seconds = int(
        duration_seconds
    )

    if duration_seconds not in ADAPTIVE_DURATIONS:
        raise ValueError(
            f"Invalid adaptive duration: {duration_seconds}"
        )


    budget = int(
        float(
            row[
                "calibration_budget_seconds_per_class"
            ]
        )
    )

    if budget not in ADAPTIVE_BUDGETS:
        raise ValueError(
            f"Invalid adaptive budget: {budget}"
        )


    test_participants = split_semicolon(
        row[
            "test_participants"
        ]
    )

    if len(
        test_participants
    ) != 1:
        raise RuntimeError(
            "Subject-adaptive row must contain one outer participant."
        )

    participant = test_participants[
        0
    ]


    cache = load_internal_cache(
        duration_seconds
    )

    path = str(
        row[
            "path"
        ]
    )

    eligible = internal_eligibility_mask(
        cache,
        path=path,
    )

    participant_values = cache[
        "participant"
    ].astype(str)

    labels = cache[
        "label"
    ].astype(
        np.int64
    )

    phases = cache[
        "phase"
    ].astype(
        np.int64
    )

    source_start = cache[
        "source_start_4s_index"
    ].astype(
        np.int64
    )

    start_seconds = (
        source_start
        * 4.0
    )


    stride_seconds = (
        duration_seconds
        * 0.5
    )

    calibration_windows = max(
        1,
        int(
            math.ceil(
                budget
                / stride_seconds
            )
        ),
    )

    calibration_raw_end = (
        duration_seconds
        + (
            calibration_windows
            - 1
        )
        * stride_seconds
    )

    test_threshold = (
        calibration_raw_end
        + duration_seconds
    )


    participant_mask_values = (
        participant_values
        == participant
    )


    calibration_indices = []
    test_indices = []

    calibration_by_class = {}
    test_by_class = {}

    raw_calibration_by_class = {}


    phase_label_pairs = (
        (
            1,
            0,
        ),
        (
            2,
            2,
        ),
        (
            3,
            1,
        )
    )


    for phase, label in phase_label_pairs:

        base = (
            participant_mask_values
            & (
                phases
                == phase
            )
            & (
                labels
                == label
            )
        )


        raw_calibration = (
            base
            & (
                (
                    start_seconds
                    + duration_seconds
                )
                <= (
                    calibration_raw_end
                    + 1.0e-9
                )
            )
        )


        if int(
            raw_calibration.sum()
        ) != calibration_windows:

            raise RuntimeError(
                f"{participant} phase {phase}, "
                f"{duration_seconds}s, budget {budget}: "
                f"expected {calibration_windows} raw calibration "
                f"windows, found {int(raw_calibration.sum())}."
            )


        calibration = (
            raw_calibration
            & eligible
        )

        test = (
            base
            & eligible
            & (
                start_seconds
                >= (
                    test_threshold
                    - 1.0e-9
                )
            )
        )


        calibration_count = int(
            calibration.sum()
        )

        test_count = int(
            test.sum()
        )


        if calibration_count <= 0:
            raise RuntimeError(
                f"{participant} class {label}, {path}, "
                f"{duration_seconds}s, budget {budget}: "
                "no eligible calibration window."
            )

        if test_count <= 0:
            raise RuntimeError(
                f"{participant} class {label}, {path}, "
                f"{duration_seconds}s, budget {budget}: "
                "no eligible test window."
            )


        raw_calibration_by_class[
            label
        ] = int(
            raw_calibration.sum()
        )

        calibration_by_class[
            label
        ] = calibration_count

        test_by_class[
            label
        ] = test_count


        calibration_indices.extend(
            np.flatnonzero(
                calibration
            ).tolist()
        )

        test_indices.extend(
            np.flatnonzero(
                test
            ).tolist()
        )


    calibration_indices = np.asarray(
        sorted(
            calibration_indices
        ),
        dtype=np.int64,
    )

    test_indices = np.asarray(
        sorted(
            test_indices
        ),
        dtype=np.int64,
    )


    if set(
        calibration_indices.tolist()
    ) & set(
        test_indices.tolist()
    ):
        raise RuntimeError(
            "Adaptive calibration/test index overlap."
        )


    assert_class_coverage(
        labels,
        calibration_indices,
        expected_classes=(
            0,
            1,
            2,
        ),
        context=(
            f"adaptive calibration {participant} "
            f"{path} {duration_seconds}s budget {budget}"
        ),
    )

    assert_class_coverage(
        labels,
        test_indices,
        expected_classes=(
            0,
            1,
            2,
        ),
        context=(
            f"adaptive test {participant} "
            f"{path} {duration_seconds}s budget {budget}"
        ),
    )


    return {
        "participant":
            participant,

        "duration_seconds":
            duration_seconds,

        "budget_seconds_per_class":
            budget,

        "stride_seconds":
            stride_seconds,

        "raw_calibration_windows_per_class":
            calibration_windows,

        "calibration_raw_end_seconds":
            calibration_raw_end,

        "guard_seconds":
            duration_seconds,

        "test_start_threshold_seconds":
            test_threshold,

        "calibration_indices":
            calibration_indices,

        "test_indices":
            test_indices,

        "raw_calibration_by_class":
            raw_calibration_by_class,

        "eligible_calibration_by_class":
            calibration_by_class,

        "eligible_test_by_class":
            test_by_class,
    }


def scan_bbbd_cache_metadata() -> pd.DataFrame:

    files = sorted(
        BBBD_CACHE_ROOT.rglob(
            "*.npz"
        )
    )

    if len(
        files
    ) != 392:
        raise RuntimeError(
            f"Expected 392 BBBD cache files, found {len(files)}."
        )


    rows = []


    for path in files:

        with np.load(
            path,
            allow_pickle=False,
        ) as archive:

            labels = archive[
                "label"
            ].astype(
                np.int64
            )

            participants = np.unique(
                archive[
                    "participant"
                ].astype(str)
            )

            recording_ids = np.unique(
                archive[
                    "recording_id"
                ].astype(str)
            )


            if len(
                participants
            ) != 1:
                raise RuntimeError(
                    f"{path.name}: expected one participant."
                )

            if len(
                recording_ids
            ) != 1:
                raise RuntimeError(
                    f"{path.name}: expected one recording ID."
                )


            unique_labels = np.unique(
                labels
            )

            if len(
                unique_labels
            ) != 1:
                raise RuntimeError(
                    f"{path.name}: expected one recording label."
                )


            participant = str(
                participants[
                    0
                ]
            )

            dataset = (
                "experiment2"
                if participant.startswith(
                    "experiment2::"
                )
                else (
                    "experiment3"
                    if participant.startswith(
                        "experiment3::"
                    )
                    else "unknown"
                )
            )


            if dataset == "unknown":
                raise RuntimeError(
                    f"Unexpected BBBD participant: {participant}"
                )


            n = len(
                labels
            )


            for modality, expected_channels in [
                (
                    "eeg",
                    8,
                ),
                (
                    "ecg",
                    1,
                ),
                (
                    "pupil",
                    1,
                ),
            ]:

                array = archive[
                    modality
                ]

                if array.shape[
                    0
                ] != n:
                    raise RuntimeError(
                        f"{path.name}: {modality} row mismatch."
                    )

                if array.shape[
                    1
                ] != expected_channels:
                    raise RuntimeError(
                        f"{path.name}: {modality} channel mismatch."
                    )

                if array.shape[
                    2
                ] != 512:
                    raise RuntimeError(
                        f"{path.name}: {modality} time mismatch."
                    )


            rows.append(
                {
                    "dataset":
                        dataset,

                    "participant":
                        participant,

                    "recording_id":
                        str(
                            recording_ids[
                                0
                            ]
                        ),

                    "label":
                        int(
                            unique_labels[
                                0
                            ]
                        ),

                    "windows":
                        int(
                            n
                        ),

                    "cache_path":
                        str(
                            path.relative_to(
                                ROOT
                            )
                        ),
                }
            )


    frame = pd.DataFrame(
        rows
    )


    if int(
        frame[
            "windows"
        ].sum()
    ) != 51831:
        raise RuntimeError(
            "BBBD cache total is not 51,831 windows."
        )


    if frame[
        "participant"
    ].nunique() != 36:
        raise RuntimeError(
            "BBBD cache does not contain 36 participants."
        )


    return frame


def bind_bbbd_row(
    row: Mapping,
    metadata: pd.DataFrame,
) -> dict:

    train = split_semicolon(
        row[
            "train_participants"
        ]
    )

    validation = split_semicolon(
        row[
            "validation_participants"
        ]
    )

    test = split_semicolon(
        row[
            "test_participants"
        ]
    )


    train_set = set(
        train
    )

    validation_set = set(
        validation
    )

    test_set = set(
        test
    )


    if train_set & validation_set:
        raise RuntimeError(
            "BBBD train/validation participant overlap."
        )

    if train_set & test_set:
        raise RuntimeError(
            "BBBD train/test participant overlap."
        )

    if validation_set & test_set:
        raise RuntimeError(
            "BBBD validation/test participant overlap."
        )


    known = set(
        metadata[
            "participant"
        ].astype(str)
    )


    for name, values in [
        (
            "train",
            train_set,
        ),
        (
            "validation",
            validation_set,
        ),
        (
            "test",
            test_set,
        ),
    ]:

        unknown = (
            values
            - known
        )

        if unknown:
            raise RuntimeError(
                f"BBBD {name} contains unknown participants: "
                f"{sorted(unknown)}"
            )


    def partition_summary(
        participants: set[str],
        name: str,
    ) -> dict:

        subset = metadata.loc[
            metadata[
                "participant"
            ].astype(str).isin(
                participants
            )
        ]

        if subset.empty:
            raise RuntimeError(
                f"BBBD {name} partition is empty."
            )


        labels = set(
            subset[
                "label"
            ].astype(int)
        )

        if labels != {
            0,
            1,
        }:
            raise RuntimeError(
                f"BBBD {name} partition lacks binary class coverage: "
                f"{sorted(labels)}"
            )


        return {
            "participants":
                int(
                    subset[
                        "participant"
                    ].nunique()
                ),

            "recordings":
                int(
                    len(
                        subset
                    )
                ),

            "windows":
                int(
                    subset[
                        "windows"
                    ].sum()
                ),

            "label_windows":
                {
                    str(
                        int(
                            label
                        )
                    ):
                        int(
                            subset.loc[
                                subset[
                                    "label"
                                ].astype(int)
                                == int(
                                    label
                                ),
                                "windows",
                            ].sum()
                        )
                    for label
                    in sorted(
                        labels
                    )
                },
        }


    return {
        "train":
            partition_summary(
                train_set,
                "train",
            ),

        "validation":
            partition_summary(
                validation_set,
                "validation",
            ),

        "test":
            partition_summary(
                test_set,
                "test",
            ),
    }


def load_internal_partition_arrays(
    *,
    duration_seconds: int,
    path: str,
    indices: np.ndarray,
) -> tuple[
    dict[str, np.ndarray],
    np.ndarray,
]:

    cache = load_internal_cache(
        duration_seconds
    )

    indices = np.asarray(
        indices,
        dtype=np.int64,
    )

    modalities = path_modalities(
        path
    )


    key_map = {
        "EEG":
            "eeg",

        "ECG":
            "ecg",

        "Pupil":
            "pupil",
    }


    arrays = {
        modality:
            np.asarray(
                cache[
                    key_map[
                        modality
                    ]
                ][
                    indices
                ],
                dtype=np.float32,
            )
        for modality
        in modalities
    }


    labels = np.asarray(
        cache[
            "label"
        ][
            indices
        ],
        dtype=np.int64,
    )


    return (
        arrays,
        labels,
    )


def load_bbbd_partition_arrays(
    *,
    participants: Sequence[str],
    path: str,
) -> tuple[
    dict[str, np.ndarray],
    np.ndarray,
    np.ndarray,
]:

    requested = set(
        str(
            value
        )
        for value
        in participants
    )

    if not requested:
        raise ValueError(
            "No BBBD participants requested."
        )


    modalities = path_modalities(
        path
    )


    key_map = {
        "EEG":
            "eeg",

        "ECG":
            "ecg",

        "Pupil":
            "pupil",
    }


    modality_parts = {
        modality:
            []
        for modality
        in modalities
    }

    labels = []
    recording_ids = []

    found = set()


    for cache_path in sorted(
        BBBD_CACHE_ROOT.rglob(
            "*.npz"
        )
    ):

        with np.load(
            cache_path,
            allow_pickle=False,
        ) as archive:

            participant_values = np.unique(
                archive[
                    "participant"
                ].astype(str)
            )

            if len(
                participant_values
            ) != 1:
                raise RuntimeError(
                    f"{cache_path.name}: invalid participant metadata."
                )

            participant = str(
                participant_values[
                    0
                ]
            )

            if participant not in requested:
                continue


            found.add(
                participant
            )


            for modality in modalities:

                modality_parts[
                    modality
                ].append(
                    np.asarray(
                        archive[
                            key_map[
                                modality
                            ]
                        ],
                        dtype=np.float32,
                    )
                )


            labels.append(
                np.asarray(
                    archive[
                        "label"
                    ],
                    dtype=np.int64,
                )
            )

            recording_ids.append(
                archive[
                    "recording_id"
                ].astype(str)
            )


    if found != requested:
        raise RuntimeError(
            "Missing BBBD participants from cache: "
            f"{sorted(requested - found)}"
        )


    arrays = {
        modality:
            np.concatenate(
                parts,
                axis=0,
            )
        for modality, parts
        in modality_parts.items()
    }


    labels_array = np.concatenate(
        labels
    )

    recording_array = np.concatenate(
        recording_ids
    )


    return (
        arrays,
        labels_array,
        recording_array,
    )
