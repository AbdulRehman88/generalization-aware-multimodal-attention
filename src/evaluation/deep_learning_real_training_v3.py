from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import inspect
import json
import math
import os
import random
import re
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize
from tqdm.auto import tqdm

from src.data_loaders.deep_learning_real_data_binding import (
    adaptive_temporal_roles,
    bind_internal_conventional_row,
    bind_internal_participant_split,
    regenerate_frozen_conventional_index_map,
)
from src.models.deep_learning_training_runtime import (
    expand_execution_registry,
)

import src.models.deep_learning_training_runtime as runtime


ROOT = Path(__file__).resolve().parents[2]

CONFIG = ROOT / "configs/deep_learning_baselines.yaml"

REGISTRY = (
    ROOT
    / "artifacts/revision/manifests/"
      "deep_learning_execution_registry_v3.csv"
)

CACHE_ROOT = (
    ROOT
    / "outputs/revision/deep_learning_cache_v1"
)

OUTPUT_ROOT = (
    ROOT
    / "outputs/revision/evaluation/deep_learning_v3"
)

AUTHORIZATION = (
    ROOT
    / "_research_audit/"
      "deep_learning_real_training_authorization_v1.json"
)

RESULTS_CSV = OUTPUT_ROOT / "operation_results.csv"

NESTED_SELECTION_CSV = (
    OUTPUT_ROOT
    / "internal_nested_loso_selected_durations.csv"
)

EXPECTED_REGISTRY_SHA256 = (
    "f95602147e4c71d3a6317a5dbe513106ea8a87e4b07b76a89d5cc3d9f40e9da4"
)

EXPECTED_OPERATIONS = 3000

PATH_MODALITIES = {
    "EEG":
        ["eeg"],

    "ECG":
        ["ecg"],

    "Pupil":
        ["pupil"],

    "ECG_EEG":
        ["ecg", "eeg"],

    "ECG_Pupil":
        ["ecg", "pupil"],

    "EEG_Pupil":
        ["eeg", "pupil"],

    "ECG_EEG_Pupil":
        ["ecg", "eeg", "pupil"],
}

INTERNAL_EXPECTED_COUNTS = {
    4:
        3315,

    8:
        3276,

    16:
        1599,

    24:
        1066,

    32:
        767,
}

GLOBAL_MAX_EPOCHS = 100
GLOBAL_PATIENCE = 15
GLOBAL_LR = 1e-3
GLOBAL_WEIGHT_DECAY = 1e-4

ADAPTIVE_MAX_EPOCHS = 20
ADAPTIVE_LR = 5e-4
ADAPTIVE_WEIGHT_DECAY = 1e-4

BATCH_SIZE = 64
GRAD_CLIP = 5.0

THRESHOLDS = [
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
]


def sha256_file(
    path: Path,
) -> str:

    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as stream:

        while True:

            block = stream.read(
                1024 * 1024
            )

            if not block:
                break

            digest.update(
                block
            )

    return digest.hexdigest()


def atomic_json(
    path: Path,
    payload: dict,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp = path.with_suffix(
        path.suffix + ".tmp"
    )

    temp.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    os.replace(
        temp,
        path,
    )


def atomic_torch_save(
    path: Path,
    payload: dict,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp = path.with_suffix(
        path.suffix + ".tmp"
    )

    torch.save(
        payload,
        temp,
    )

    os.replace(
        temp,
        path,
    )


def safe_name(
    value: str,
) -> str:

    value = re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        str(
            value
        ),
    )

    return value.strip(
        "_"
    )


def operation_directory(
    operation: dict,
) -> Path:

    operation_id = str(
        operation[
            "operation_id"
        ]
    )

    short_hash = hashlib.sha256(
        operation_id.encode(
            "utf-8"
        )
    ).hexdigest()[
        :12
    ]

    label = "_".join(
        [
            safe_name(
                str(
                    operation[
                        "split_id"
                    ]
                )
            ),

            safe_name(
                str(
                    operation[
                        "model"
                    ]
                )
            ),

            safe_name(
                str(
                    operation[
                        "path"
                    ]
                )
            ),

            f"seed{int(operation['seed'])}",
        ]
    )

    duration = operation.get(
        "duration_candidate_seconds"
    )

    if (
        (
            duration is None
            or pd.isna(
                duration
            )
        )
        and str(
            operation.get(
                "stage",
                "",
            )
        )
        != "internal_subject_adaptive"
    ):

        fallback_duration = operation.get(
            "duration_seconds"
        )

        if (
            fallback_duration is not None
            and not pd.isna(
                fallback_duration
            )
        ):

            duration = fallback_duration

    if (
        duration is not None
        and not pd.isna(
            duration
        )
    ):

        label += (
            f"_d{int(float(duration))}"
        )

    budget = operation.get(
        "calibration_budget_seconds_per_class"
    )

    if (
        budget is not None
        and not pd.isna(
            budget
        )
    ):

        label += (
            f"_b{int(float(budget))}"
        )

    return (
        OUTPUT_ROOT
        / "operations"
        / safe_name(
            str(
                operation[
                    "stage"
                ]
            )
        )
        / f"{short_hash}_{label}"
    )


def set_determinism(
    seed: int,
) -> None:

    os.environ[
        "PYTHONHASHSEED"
    ] = str(
        seed
    )

    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            seed
        )

    torch.use_deterministic_algorithms(
        True,
        warn_only=True,
    )

    if hasattr(
        torch.backends,
        "cudnn",
    ):

        torch.backends.cudnn.benchmark = False

        torch.backends.cudnn.deterministic = True


def parse_people(
    value: Any,
) -> list[str]:

    if value is None:
        return []

    if isinstance(
        value,
        float,
    ) and math.isnan(
        value
    ):
        return []

    if isinstance(
        value,
        (
            list,
            tuple,
            np.ndarray,
        ),
    ):
        return [
            str(
                item
            ).strip()
            for item in value
            if str(
                item
            ).strip()
        ]

    text = str(
        value
    ).strip()

    if not text:
        return []

    if (
        text.startswith(
            "["
        )
        and text.endswith(
            "]"
        )
    ):

        try:

            parsed = ast.literal_eval(
                text
            )

            return parse_people(
                parsed
            )

        except Exception:
            pass

    return [
        token.strip()
        for token
        in re.split(
            r"[;,|]+",
            text,
        )
        if token.strip()
    ]


def scalar_text(
    value: np.ndarray | Any,
) -> str:

    array = np.asarray(
        value
    )

    if array.ndim == 0:

        item = array.item()

    else:

        item = array.reshape(
            -1
        )[
            0
        ]

    if isinstance(
        item,
        bytes,
    ):

        return item.decode(
            "utf-8"
        )

    return str(
        item
    )


def canonical_participant(
    raw: str,
    source_path: Path,
    experiment_hint: str = "",
) -> str:

    value = str(
        raw
    ).strip()

    if "::" in value:
        return value

    searchable = (
        str(
            source_path
        )
        + " "
        + str(
            experiment_hint
        )
    ).lower()

    if (
        "experiment2"
        in searchable
        or "experiment_2"
        in searchable
        or "exp2"
        in searchable
    ):

        return (
            f"experiment2::{value}"
        )

    if (
        "experiment3"
        in searchable
        or "experiment_3"
        in searchable
        or "exp3"
        in searchable
    ):

        return (
            f"experiment3::{value}"
        )

    return value


def canonical_signal(
    array: np.ndarray,
    modality: str,
    n_samples: int,
) -> np.ndarray:

    value = np.asarray(
        array
    )

    if value.shape[
        0
    ] != n_samples:

        raise RuntimeError(
            f"{modality}: first dimension {value.shape[0]} "
            f"does not match labels {n_samples}."
        )

    if value.ndim == 2:

        value = value[
            :,
            None,
            :,
        ]

    elif value.ndim == 3:

        # N x T x C -> N x C x T
        if (
            value.shape[
                1
            ]
            > 32
            and value.shape[
                2
            ]
            <= 16
        ):

            value = np.transpose(
                value,
                (
                    0,
                    2,
                    1,
                ),
            )

    else:

        raise RuntimeError(
            f"{modality}: expected 2D/3D temporal signal; "
            f"received shape {value.shape}."
        )

    return np.asarray(
        value,
        dtype=np.float32,
    )


def first_key(
    archive,
    candidates: list[str],
) -> str | None:

    lower_map = {
        str(
            key
        ).lower():
            str(
                key
            )
        for key
        in archive.files
    }

    for candidate in candidates:

        if candidate.lower() in lower_map:

            return lower_map[
                candidate.lower()
            ]

    return None


def signal_from_archive(
    archive,
    modality: str,
    n_samples: int,
) -> np.ndarray:

    candidates = {
        "eeg":
            [
                "eeg",
                "EEG",
            ],

        "ecg":
            [
                "ecg",
                "ECG",
            ],

        "pupil":
            [
                "pupil",
                "pupil_size",
                "Pupil",
                "PUPIL",
            ],
    }[
        modality
    ]

    key = first_key(
        archive,
        candidates,
    )

    if key is None:

        raise RuntimeError(
            f"Missing {modality} signal in NPZ with keys "
            f"{archive.files}."
        )

    return canonical_signal(
        archive[
            key
        ],
        modality,
        n_samples,
    )


def discover_internal_caches() -> dict[int, Path]:

    if not CACHE_ROOT.exists():

        raise FileNotFoundError(
            CACHE_ROOT
        )

    matches: dict[
        int,
        list[
            Path
        ],
    ] = {
        duration:
            []
        for duration
        in INTERNAL_EXPECTED_COUNTS
    }

    for path in CACHE_ROOT.rglob(
        "*.npz"
    ):

        try:

            with np.load(
                path,
                allow_pickle=True,
            ) as archive:

                label_key = first_key(
                    archive,
                    [
                        "label",
                        "labels",
                        "y",
                    ],
                )

                participant_key = first_key(
                    archive,
                    [
                        "participant",
                        "participants",
                    ],
                )

                eeg_key = first_key(
                    archive,
                    [
                        "eeg",
                        "EEG",
                    ],
                )

                ecg_key = first_key(
                    archive,
                    [
                        "ecg",
                        "ECG",
                    ],
                )

                pupil_key = first_key(
                    archive,
                    [
                        "pupil",
                        "pupil_size",
                        "Pupil",
                    ],
                )

                if (
                    label_key is None
                    or participant_key is None
                    or eeg_key is None
                    or ecg_key is None
                    or pupil_key is None
                ):

                    continue

                labels = np.asarray(
                    archive[
                        label_key
                    ]
                ).reshape(
                    -1
                )

                n = len(
                    labels
                )

                duration = None

                window_key = first_key(
                    archive,
                    [
                        "window_seconds",
                        "duration_seconds",
                    ],
                )

                if window_key is not None:

                    raw_window = np.asarray(
                        archive[
                            window_key
                        ]
                    ).reshape(
                        -1
                    )

                    if len(
                        raw_window
                    ):

                        duration = int(
                            round(
                                float(
                                    raw_window[
                                        0
                                    ]
                                )
                            )
                        )

                if duration is None:

                    eeg_shape = archive[
                        eeg_key
                    ].shape

                    temporal = int(
                        max(
                            eeg_shape[
                                1:
                            ]
                        )
                    )

                    duration = int(
                        round(
                            temporal
                            / 128.0
                        )
                    )

                if (
                    duration
                    in INTERNAL_EXPECTED_COUNTS
                    and n
                    == INTERNAL_EXPECTED_COUNTS[
                        duration
                    ]
                ):

                    matches[
                        duration
                    ].append(
                        path
                    )

        except Exception:
            continue

    resolved = {}

    for duration, paths in matches.items():

        if len(
            paths
        ) != 1:

            raise RuntimeError(
                f"Internal {duration}s cache discovery expected exactly "
                f"one aggregate NPZ; found {len(paths)}: "
                f"{[str(p) for p in paths]}"
            )

        resolved[
            duration
        ] = paths[
            0
        ]

    return resolved


INTERNAL_CACHE_MAP: dict[
    int,
    Path
] = {}


@lru_cache(
    maxsize=5
)
def load_internal_cache(
    duration: int,
) -> dict[str, Any]:

    path = INTERNAL_CACHE_MAP[
        int(
            duration
        )
    ]

    with np.load(
        path,
        allow_pickle=True,
    ) as archive:

        label_key = first_key(
            archive,
            [
                "label",
                "labels",
                "y",
            ],
        )

        participant_key = first_key(
            archive,
            [
                "participant",
                "participants",
            ],
        )

        if (
            label_key is None
            or participant_key is None
        ):

            raise RuntimeError(
                f"Internal cache missing labels/participants: {path}"
            )

        labels = np.asarray(
            archive[
                label_key
            ],
            dtype=np.int64,
        ).reshape(
            -1
        )

        n = len(
            labels
        )

        participants_raw = np.asarray(
            archive[
                participant_key
            ]
        )

        if participants_raw.ndim == 0:

            participants = np.repeat(
                scalar_text(
                    participants_raw
                ),
                n,
            )

        else:

            participants = participants_raw.astype(
                str
            ).reshape(
                -1
            )

        payload = {
            "eeg":
                signal_from_archive(
                    archive,
                    "eeg",
                    n,
                ),

            "ecg":
                signal_from_archive(
                    archive,
                    "ecg",
                    n,
                ),

            "pupil":
                signal_from_archive(
                    archive,
                    "pupil",
                    n,
                ),

            "labels":
                labels,

            "participant":
                np.asarray(
                    participants,
                    dtype=object,
                ),

            "recording":
                np.asarray(
                    [
                        f"{participant}::internal"
                        for participant
                        in participants
                    ],
                    dtype=object,
                ),
        }

    return payload


def subset_data(
    data: dict[str, Any],
    indices: np.ndarray | list[int],
    path: str,
) -> dict[str, Any]:

    indices = np.asarray(
        indices,
        dtype=np.int64,
    )

    output = {
        "labels":
            data[
                "labels"
            ][
                indices
            ],

        "participant":
            data[
                "participant"
            ][
                indices
            ],

        "recording":
            data[
                "recording"
            ][
                indices
            ],
    }

    for modality in PATH_MODALITIES[
        path
    ]:

        output[
            modality
        ] = data[
            modality
        ][
            indices
        ]

    return output


def internal_partition_data(
    duration: int,
    indices,
    path: str,
) -> dict[str, Any]:

    return subset_data(
        load_internal_cache(
            int(
                duration
            )
        ),
        indices,
        path,
    )


BBBD_FILE_META: pd.DataFrame | None = None


def discover_bbbd_files(
    registry: pd.DataFrame,
) -> pd.DataFrame:

    registry_people = set()

    bbbd_rows = registry.loc[
        registry[
            "stage"
        ].astype(str).isin(
            [
                "bbbd_within_experiment",
                "bbbd_cross_experiment",
            ]
        )
    ]

    for column in [
        "train_participants",
        "validation_participants",
        "test_participants",
    ]:

        for value in bbbd_rows[
            column
        ].tolist():

            registry_people.update(
                parse_people(
                    value
                )
            )

    rows = []

    internal_paths = set(
        INTERNAL_CACHE_MAP.values()
    )

    for path in CACHE_ROOT.rglob(
        "*.npz"
    ):

        if path in internal_paths:
            continue

        try:

            with np.load(
                path,
                allow_pickle=True,
            ) as archive:

                label_key = first_key(
                    archive,
                    [
                        "label",
                        "labels",
                        "y",
                    ],
                )

                participant_key = first_key(
                    archive,
                    [
                        "participant",
                        "participants",
                    ],
                )

                recording_key = first_key(
                    archive,
                    [
                        "recording",
                        "recording_id",
                        "recording_name",
                    ],
                )

                eeg_key = first_key(
                    archive,
                    [
                        "eeg",
                        "EEG",
                    ],
                )

                ecg_key = first_key(
                    archive,
                    [
                        "ecg",
                        "ECG",
                    ],
                )

                if (
                    label_key is None
                    or participant_key is None
                    or eeg_key is None
                    or ecg_key is None
                ):

                    continue

                labels = np.asarray(
                    archive[
                        label_key
                    ]
                ).reshape(
                    -1
                )

                if not len(
                    labels
                ):
                    continue

                raw_participant = scalar_text(
                    archive[
                        participant_key
                    ]
                )

                experiment_key = first_key(
                    archive,
                    [
                        "experiment",
                        "experiment_id",
                        "dataset",
                    ],
                )

                experiment = (
                    scalar_text(
                        archive[
                            experiment_key
                        ]
                    )
                    if experiment_key is not None
                    else ""
                )

                participant = canonical_participant(
                    raw_participant,
                    path,
                    experiment,
                )

                recording = (
                    scalar_text(
                        archive[
                            recording_key
                        ]
                    )
                    if recording_key is not None
                    else path.stem
                )

                rows.append(
                    {
                        "path":
                            path,

                        "participant":
                            participant,

                        "recording":
                            recording,

                        "windows":
                            len(
                                labels
                            ),
                    }
                )

        except Exception:
            continue

    frame = pd.DataFrame(
        rows
    )

    if len(
        frame
    ) != 392:

        raise RuntimeError(
            f"Expected 392 BBBD recording-cache files; "
            f"found {len(frame)}."
        )

    if frame[
        "participant"
    ].nunique() != 36:

        raise RuntimeError(
            "Expected 36 namespaced BBBD participants; "
            f"found {frame['participant'].nunique()}."
        )

    missing = sorted(
        registry_people
        - set(
            frame[
                "participant"
            ].astype(
                str
            )
        )
    )

    if missing:

        raise RuntimeError(
            "BBBD registry participants were not found in cache. "
            f"First missing IDs: {missing[:20]}"
        )

    return frame


@lru_cache(
    maxsize=96
)
def load_bbbd_recording(
    path_text: str,
) -> dict[str, Any]:

    path = Path(
        path_text
    )

    with np.load(
        path,
        allow_pickle=True,
    ) as archive:

        label_key = first_key(
            archive,
            [
                "label",
                "labels",
                "y",
            ],
        )

        if label_key is None:

            raise RuntimeError(
                f"BBBD cache missing label: {path}"
            )

        labels = np.asarray(
            archive[
                label_key
            ],
            dtype=np.int64,
        ).reshape(
            -1
        )

        n = len(
            labels
        )

        participant_key = first_key(
            archive,
            [
                "participant",
                "participants",
            ],
        )

        experiment_key = first_key(
            archive,
            [
                "experiment",
                "experiment_id",
                "dataset",
            ],
        )

        recording_key = first_key(
            archive,
            [
                "recording",
                "recording_id",
                "recording_name",
            ],
        )

        participant = canonical_participant(
            scalar_text(
                archive[
                    participant_key
                ]
            ),
            path,
            (
                scalar_text(
                    archive[
                        experiment_key
                    ]
                )
                if experiment_key is not None
                else ""
            ),
        )

        recording = (
            scalar_text(
                archive[
                    recording_key
                ]
            )
            if recording_key is not None
            else path.stem
        )

        payload = {
            "eeg":
                signal_from_archive(
                    archive,
                    "eeg",
                    n,
                ),

            "ecg":
                signal_from_archive(
                    archive,
                    "ecg",
                    n,
                ),

            "labels":
                labels,

            "participant":
                np.repeat(
                    participant,
                    n,
                ).astype(
                    object
                ),

            "recording":
                np.repeat(
                    recording,
                    n,
                ).astype(
                    object
                ),
        }

        pupil_key = first_key(
            archive,
            [
                "pupil",
                "pupil_size",
                "Pupil",
            ],
        )

        if pupil_key is not None:

            payload[
                "pupil"
            ] = canonical_signal(
                archive[
                    pupil_key
                ],
                "pupil",
                n,
            )

        valid_key = first_key(
            archive,
            [
                "pupil_valid",
                "pupil_available",
                "pupil_window_valid",
            ],
        )

        if valid_key is not None:

            valid = np.asarray(
                archive[
                    valid_key
                ]
            ).reshape(
                -1
            ).astype(
                bool
            )

            if len(
                valid
            ) == n:

                payload[
                    "pupil_valid"
                ] = valid

    return payload


def concatenate_data(
    parts: list[dict[str, Any]],
    path: str,
) -> dict[str, Any]:

    if not parts:

        raise RuntimeError(
            f"No data parts for path {path}."
        )

    output = {
        "labels":
            np.concatenate(
                [
                    part[
                        "labels"
                    ]
                    for part
                    in parts
                ],
                axis=0,
            ),

        "participant":
            np.concatenate(
                [
                    part[
                        "participant"
                    ]
                    for part
                    in parts
                ],
                axis=0,
            ),

        "recording":
            np.concatenate(
                [
                    part[
                        "recording"
                    ]
                    for part
                    in parts
                ],
                axis=0,
            ),
    }

    for modality in PATH_MODALITIES[
        path
    ]:

        output[
            modality
        ] = np.concatenate(
            [
                part[
                    modality
                ]
                for part
                in parts
            ],
            axis=0,
        )

    return output


def load_bbbd_partition(
    participants: list[str],
    path: str,
) -> dict[str, Any]:

    if BBBD_FILE_META is None:

        raise RuntimeError(
            "BBBD metadata was not initialized."
        )

    participant_set = set(
        participants
    )

    selected = BBBD_FILE_META.loc[
        BBBD_FILE_META[
            "participant"
        ].astype(str).isin(
            participant_set
        )
    ]

    if selected[
        "participant"
    ].nunique() != len(
        participant_set
    ):

        observed = set(
            selected[
                "participant"
            ].astype(str)
        )

        raise RuntimeError(
            "BBBD participant binding mismatch. "
            f"Missing={sorted(participant_set - observed)}"
        )

    parts = []

    requires_pupil = (
        "pupil"
        in PATH_MODALITIES[
            path
        ]
    )

    for row in selected.itertuples(
        index=False
    ):

        raw = load_bbbd_recording(
            str(
                row.path
            )
        )

        n = len(
            raw[
                "labels"
            ]
        )

        mask = np.ones(
            n,
            dtype=bool,
        )

        if requires_pupil:

            if "pupil" not in raw:

                raise RuntimeError(
                    f"Pupil path {path} requested but cache has no pupil: "
                    f"{row.path}"
                )

            if "pupil_valid" in raw:

                mask &= raw[
                    "pupil_valid"
                ]

        indices = np.flatnonzero(
            mask
        )

        part = {
            "labels":
                raw[
                    "labels"
                ][
                    indices
                ],

            "participant":
                raw[
                    "participant"
                ][
                    indices
                ],

            "recording":
                raw[
                    "recording"
                ][
                    indices
                ],
        }

        for modality in PATH_MODALITIES[
            path
        ]:

            part[
                modality
            ] = raw[
                modality
            ][
                indices
            ]

        parts.append(
            part
        )

    result = concatenate_data(
        parts,
        path,
    )

    if not len(
        result[
            "labels"
        ]
    ):

        raise RuntimeError(
            f"BBBD partition became empty for path {path}."
        )

    return result


def fit_normalizer(
    train_data: dict[str, Any],
    path: str,
) -> dict[str, dict[str, list[float]]]:

    normalizer = {}

    for modality in PATH_MODALITIES[
        path
    ]:

        array = np.asarray(
            train_data[
                modality
            ],
            dtype=np.float32,
        )

        mean = np.mean(
            array,
            axis=(
                0,
                2,
            ),
            dtype=np.float64,
        )

        std = np.std(
            array,
            axis=(
                0,
                2,
            ),
            dtype=np.float64,
        )

        std = np.where(
            std
            > 0.0,
            std,
            1.0,
        )

        normalizer[
            modality
        ] = {
            "mean":
                mean.astype(
                    float
                ).tolist(),

            "std":
                std.astype(
                    float
                ).tolist(),
        }

    return normalizer


def apply_normalizer(
    data: dict[str, Any],
    path: str,
    normalizer: dict,
) -> dict[str, Any]:

    output = {
        "labels":
            np.asarray(
                data[
                    "labels"
                ],
                dtype=np.int64,
            ),

        "participant":
            np.asarray(
                data[
                    "participant"
                ],
                dtype=object,
            ),

        "recording":
            np.asarray(
                data[
                    "recording"
                ],
                dtype=object,
            ),
    }

    for modality in PATH_MODALITIES[
        path
    ]:

        array = np.asarray(
            data[
                modality
            ],
            dtype=np.float32,
        )

        mean = np.asarray(
            normalizer[
                modality
            ][
                "mean"
            ],
            dtype=np.float32,
        )[
            None,
            :,
            None,
        ]

        std = np.asarray(
            normalizer[
                modality
            ][
                "std"
            ],
            dtype=np.float32,
        )[
            None,
            :,
            None,
        ]

        output[
            modality
        ] = (
            array
            - mean
        ) / std

    return output


def invoke_builder(
    builder,
    *,
    model_name: str,
    path: str,
    n_classes: int,
    duration: int,
    seed: int,
    device: torch.device,
):

    modalities = PATH_MODALITIES[
        path
    ]

    channel_map = {
        "eeg":
            8,

        "ecg":
            1,

        "pupil":
            1,
    }

    temporal_map = {
        "eeg":
            int(
                duration
                * 128
            ),

        "ecg":
            int(
                duration
                * 128
            ),

        "pupil":
            int(
                duration
                * (
                    30
                    if duration
                    in INTERNAL_EXPECTED_COUNTS
                    else 128
                )
            ),
    }

    protocol = yaml.safe_load(
        CONFIG.read_text(
            encoding="utf-8"
        )
    )

    pool = {
        "model_name":
            model_name,

        "name":
            model_name,

        "model":
            model_name,

        "path":
            path,

        "modality_path":
            path,

        "modalities":
            modalities,

        "n_classes":
            n_classes,

        "num_classes":
            n_classes,

        "classes":
            n_classes,

        "duration_seconds":
            duration,

        "window_seconds":
            duration,

        "duration":
            duration,

        "seed":
            seed,

        "training_seed":
            seed,

        "random_seed":
            seed,

        "device":
            device,

        "protocol":
            protocol,

        "config":
            protocol,

        "input_channels":
            {
                modality:
                    channel_map[
                        modality
                    ]
                for modality
                in modalities
            },

        "modality_channels":
            {
                modality:
                    channel_map[
                        modality
                    ]
                for modality
                in modalities
            },

        "input_lengths":
            {
                modality:
                    temporal_map[
                        modality
                    ]
                for modality
                in modalities
            },

        "sampling_rates":
            {
                modality:
                    (
                        30
                        if modality
                        == "pupil"
                        else 128
                    )
                for modality
                in modalities
            },

        "n_channels":
            (
                8
                if model_name
                == "ShallowConvNet"
                else channel_map[
                    modalities[
                        0
                    ]
                ]
            ),

        "in_channels":
            (
                8
                if model_name
                == "ShallowConvNet"
                else channel_map[
                    modalities[
                        0
                    ]
                ]
            ),

        "n_times":
            int(
                duration
                * 128
            ),

        "input_time_length":
            int(
                duration
                * 128
            ),
    }

    signature = inspect.signature(
        builder
    )

    kwargs = {}

    missing = []

    for name, parameter in signature.parameters.items():

        if name in pool:

            kwargs[
                name
            ] = pool[
                name
            ]

        elif (
            parameter.default
            is inspect.Parameter.empty
            and parameter.kind
            not in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            )
        ):

            missing.append(
                name
            )

    if missing:

        raise RuntimeError(
            "Frozen model builder has required parameters not handled "
            f"by launcher: {missing}. Signature={signature}"
        )

    built = builder(
        **kwargs
    )

    if isinstance(
        built,
        nn.Module,
    ):

        model = built

    elif (
        isinstance(
            built,
            tuple,
        )
        and built
        and isinstance(
            built[
                0
            ],
            nn.Module,
        )
    ):

        model = built[
            0
        ]

    elif (
        isinstance(
            built,
            dict,
        )
        and isinstance(
            built.get(
                "model"
            ),
            nn.Module,
        )
    ):

        model = built[
            "model"
        ]

    else:

        raise RuntimeError(
            "Frozen model builder did not return a torch.nn.Module. "
            f"Received {type(built)}."
        )

    return model.to(
        device
    )


def build_model(
    *,
    model_name: str,
    path: str,
    n_classes: int,
    duration: int,
    seed: int,
    device: torch.device,
) -> nn.Module:

    set_determinism(
        seed
    )

    builder = getattr(
        runtime,
        "build_seeded_frozen_model",
        None,
    )

    if builder is None:

        raise RuntimeError(
            "Frozen runtime no longer exposes "
            "build_seeded_frozen_model()."
        )

    return invoke_builder(
        builder,
        model_name=model_name,
        path=path,
        n_classes=n_classes,
        duration=duration,
        seed=seed,
        device=device,
    )


def configure_head_only(
    model: nn.Module,
    model_name: str,
) -> None:

    helper = getattr(
        runtime,
        "configure_final_head_only",
        None,
    )

    if helper is not None:

        signature = inspect.signature(
            helper
        )

        kwargs = {}

        for name, parameter in signature.parameters.items():

            if name == "model":

                kwargs[
                    name
                ] = model

            elif name in (
                "model_name",
                "name",
            ):

                kwargs[
                    name
                ] = model_name

            elif (
                parameter.default
                is inspect.Parameter.empty
                and parameter.kind
                not in (
                    inspect.Parameter.VAR_POSITIONAL,
                    inspect.Parameter.VAR_KEYWORD,
                )
            ):

                kwargs = {}
                break

        if kwargs:

            helper(
                **kwargs
            )

            trainable = [
                name
                for name, parameter
                in model.named_parameters()
                if parameter.requires_grad
            ]

            if not trainable:

                raise RuntimeError(
                    "Frozen final-head helper left no trainable parameter."
                )

            return

    expected_prefix = (
        "classifier"
        if model_name
        == "ShallowConvNet"
        else "fusion.3"
    )

    for name, parameter in model.named_parameters():

        parameter.requires_grad = name.startswith(
            expected_prefix
        )

    trainable = [
        name
        for name, parameter
        in model.named_parameters()
        if parameter.requires_grad
    ]

    if not trainable:

        raise RuntimeError(
            f"Could not identify final head {expected_prefix}."
        )


def move_batch(
    data: dict[str, Any],
    indices: np.ndarray,
    path: str,
    device: torch.device,
) -> dict[str, torch.Tensor]:

    return {
        modality:
            torch.as_tensor(
                data[
                    modality
                ][
                    indices
                ],
                dtype=torch.float32,
                device=device,
            )
        for modality
        in PATH_MODALITIES[
            path
        ]
    }


def forward_model(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    model_name: str,
    path: str,
) -> torch.Tensor:

    cached_mode = getattr(
        model,
        "_real_launcher_forward_mode",
        None,
    )

    modalities = PATH_MODALITIES[
        path
    ]

    def execute(
        mode: str,
    ):

        if mode == "single":

            return model(
                batch[
                    modalities[
                        0
                    ]
                ]
            )

        if mode == "single_4d":

            return model(
                batch[
                    modalities[
                        0
                    ]
                ].unsqueeze(
                    1
                )
            )

        if mode == "dict_canonical":

            canonical_names = {
                "eeg":
                    "EEG",

                "ecg":
                    "ECG",

                "pupil":
                    "Pupil",
            }

            return model(
                {
                    canonical_names[
                        key
                    ]:
                        batch[
                            key
                        ]
                    for key
                    in modalities
                }
            )

        if mode == "dict_lower":

            return model(
                {
                    key:
                        batch[
                            key
                        ]
                    for key
                    in modalities
                }
            )

        if mode == "dict_upper":

            return model(
                {
                    key.upper():
                        batch[
                            key
                        ]
                    for key
                    in modalities
                }
            )

        if mode == "kwargs":

            return model(
                **{
                    key:
                        batch[
                            key
                        ]
                    for key
                    in modalities
                }
            )

        if mode == "positional":

            return model(
                *[
                    batch[
                        key
                    ]
                    for key
                    in modalities
                ]
            )

        raise RuntimeError(
            mode
        )

    if cached_mode is not None:

        output = execute(
            cached_mode
        )

    else:

        modes = (
            [
                "single",
                "single_4d",
            ]
            if model_name
            == "ShallowConvNet"
            else [
                "dict_canonical",
                "dict_lower",
                "dict_upper",
                "kwargs",
                "positional",
            ]
        )

        errors = []

        output = None

        for mode in modes:

            try:

                candidate = execute(
                    mode
                )

                if (
                    torch.is_tensor(
                        candidate
                    )
                    and candidate.ndim
                    == 2
                ):

                    output = candidate

                    setattr(
                        model,
                        "_real_launcher_forward_mode",
                        mode,
                    )

                    break

            except Exception as exc:

                errors.append(
                    (
                        mode,
                        repr(
                            exc
                        ),
                    )
                )

        if output is None:

            raise RuntimeError(
                f"Could not resolve forward interface for "
                f"{model_name}/{path}. Errors={errors}"
            )

    if not torch.is_tensor(
        output
    ):

        if (
            isinstance(
                output,
                (
                    tuple,
                    list,
                ),
            )
            and output
            and torch.is_tensor(
                output[
                    0
                ]
            )
        ):

            output = output[
                0
            ]

        elif (
            isinstance(
                output,
                dict,
            )
        ):

            for key in [
                "logits",
                "output",
                "predictions",
            ]:

                if (
                    key in output
                    and torch.is_tensor(
                        output[
                            key
                        ]
                    )
                ):

                    output = output[
                        key
                    ]

                    break

    if (
        not torch.is_tensor(
            output
        )
        or output.ndim
        != 2
    ):

        raise RuntimeError(
            f"Model output must be [batch, classes]; "
            f"received {type(output)} / "
            f"{getattr(output, 'shape', None)}."
        )

    return output


def class_weights(
    labels: np.ndarray,
    n_classes: int,
    device: torch.device,
) -> torch.Tensor:

    counts = np.bincount(
        labels.astype(
            int
        ),
        minlength=n_classes,
    ).astype(
        np.float64
    )

    if np.any(
        counts
        <= 0
    ):

        raise RuntimeError(
            f"Training partition lacks a class: {counts.tolist()}"
        )

    weights = (
        len(
            labels
        )
        / (
            n_classes
            * counts
        )
    )

    return torch.as_tensor(
        weights,
        dtype=torch.float32,
        device=device,
    )


def metric_bundle(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float]:

    labels = np.asarray(
        labels,
        dtype=int,
    )

    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    predictions = np.argmax(
        probabilities,
        axis=1,
    )

    metrics = {
        "accuracy":
            float(
                accuracy_score(
                    labels,
                    predictions,
                )
            ),

        "balanced_accuracy":
            float(
                balanced_accuracy_score(
                    labels,
                    predictions,
                )
            ),

        "macro_f1":
            float(
                f1_score(
                    labels,
                    predictions,
                    average="macro",
                    zero_division=0,
                )
            ),
    }

    try:

        if probabilities.shape[
            1
        ] == 2:

            metrics[
                "roc_auc"
            ] = float(
                roc_auc_score(
                    labels,
                    probabilities[
                        :,
                        1
                    ],
                )
            )

            metrics[
                "pr_auc"
            ] = float(
                average_precision_score(
                    labels,
                    probabilities[
                        :,
                        1
                    ],
                )
            )

        else:

            classes = np.arange(
                probabilities.shape[
                    1
                ]
            )

            binary = label_binarize(
                labels,
                classes=classes,
            )

            metrics[
                "roc_auc"
            ] = float(
                roc_auc_score(
                    binary,
                    probabilities,
                    average="macro",
                    multi_class="ovr",
                )
            )

            metrics[
                "pr_auc"
            ] = float(
                average_precision_score(
                    binary,
                    probabilities,
                    average="macro",
                )
            )

    except Exception:

        metrics[
            "roc_auc"
        ] = float(
            "nan"
        )

        metrics[
            "pr_auc"
        ] = float(
            "nan"
        )

    return metrics


def binary_metrics_at_threshold(
    labels: np.ndarray,
    probabilities_positive: np.ndarray,
    threshold: float,
) -> dict[str, float]:

    predictions = (
        probabilities_positive
        >= float(
            threshold
        )
    ).astype(
        int
    )

    return {
        "accuracy":
            float(
                accuracy_score(
                    labels,
                    predictions,
                )
            ),

        "balanced_accuracy":
            float(
                balanced_accuracy_score(
                    labels,
                    predictions,
                )
            ),

        "macro_f1":
            float(
                f1_score(
                    labels,
                    predictions,
                    average="macro",
                    zero_division=0,
                )
            ),

        "roc_auc":
            float(
                roc_auc_score(
                    labels,
                    probabilities_positive,
                )
            ),

        "pr_auc":
            float(
                average_precision_score(
                    labels,
                    probabilities_positive,
                )
            ),
    }


def predict(
    model: nn.Module,
    data: dict[str, Any],
    path: str,
    model_name: str,
    device: torch.device,
) -> tuple[np.ndarray, float]:

    model.eval()

    probabilities = []

    start = time.perf_counter()

    if device.type == "cuda":

        torch.cuda.synchronize()

    with torch.no_grad():

        for start_index in range(
            0,
            len(
                data[
                    "labels"
                ]
            ),
            BATCH_SIZE,
        ):

            indices = np.arange(
                start_index,
                min(
                    start_index
                    + BATCH_SIZE,
                    len(
                        data[
                            "labels"
                        ]
                    ),
                ),
                dtype=np.int64,
            )

            batch = move_batch(
                data,
                indices,
                path,
                device,
            )

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=(
                    device.type
                    == "cuda"
                ),
            ):

                logits = forward_model(
                    model,
                    batch,
                    model_name,
                    path,
                )

            probabilities.append(
                torch.softmax(
                    logits,
                    dim=1,
                )
                .detach()
                .cpu()
                .numpy()
            )

    if device.type == "cuda":

        torch.cuda.synchronize()

    elapsed = (
        time.perf_counter()
        - start
    )

    return (
        np.concatenate(
            probabilities,
            axis=0,
        ),
        elapsed,
    )


def aggregate_recordings(
    data: dict[str, Any],
    probabilities: np.ndarray,
) -> pd.DataFrame:

    frame = pd.DataFrame(
        {
            "participant":
                data[
                    "participant"
                ].astype(
                    str
                ),

            "recording":
                data[
                    "recording"
                ].astype(
                    str
                ),

            "label":
                data[
                    "labels"
                ].astype(
                    int
                ),
        }
    )

    for class_index in range(
        probabilities.shape[
            1
        ]
    ):

        frame[
            f"p{class_index}"
        ] = probabilities[
            :,
            class_index
        ]

    aggregations = {
        "participant":
            "first",

        "label":
            "first",
    }

    for class_index in range(
        probabilities.shape[
            1
        ]
    ):

        aggregations[
            f"p{class_index}"
        ] = "mean"

    return (
        frame.groupby(
            "recording",
            as_index=False,
        )
        .agg(
            aggregations
        )
    )


def choose_threshold(
    recording_frame: pd.DataFrame,
) -> tuple[float, dict[str, float]]:

    labels = recording_frame[
        "label"
    ].to_numpy(
        dtype=int
    )

    positive = recording_frame[
        "p1"
    ].to_numpy(
        dtype=float
    )

    candidates = []

    for threshold in THRESHOLDS:

        metrics = binary_metrics_at_threshold(
            labels,
            positive,
            threshold,
        )

        candidates.append(
            (
                metrics[
                    "balanced_accuracy"
                ],
                metrics[
                    "macro_f1"
                ],
                -abs(
                    threshold
                    - 0.5
                ),
                threshold,
                metrics,
            )
        )

    candidates.sort(
        reverse=True,
        key=lambda item:
            (
                item[
                    0
                ],
                item[
                    1
                ],
                item[
                    2
                ],
            ),
    )

    best = candidates[
        0
    ]

    return (
        float(
            best[
                3
            ]
        ),
        best[
            4
        ],
    )


def recording_sampling_weights(
    recording_ids: np.ndarray,
) -> np.ndarray:

    series = pd.Series(
        recording_ids.astype(
            str
        )
    )

    counts = series.value_counts()

    weights = series.map(
        lambda value:
            1.0
            / float(
                counts[
                    value
                ]
            )
    ).to_numpy(
        dtype=np.float64
    )

    weights /= np.mean(
        weights
    )

    return weights


def epoch_indices(
    n: int,
    seed: int,
    epoch: int,
    sampling_weights: np.ndarray | None,
) -> np.ndarray:

    generator = torch.Generator(
        device="cpu"
    )

    generator.manual_seed(
        int(
            seed
            * 100003
            + epoch
            * 1009
            + 17
        )
    )

    if sampling_weights is None:

        return (
            torch.randperm(
                n,
                generator=generator,
            )
            .numpy()
            .astype(
                np.int64
            )
        )

    weights = torch.as_tensor(
        sampling_weights,
        dtype=torch.double,
    )

    return (
        torch.multinomial(
            weights,
            num_samples=n,
            replacement=True,
            generator=generator,
        )
        .numpy()
        .astype(
            np.int64
        )
    )


def checkpoint_payload(
    *,
    operation: dict,
    model: nn.Module,
    optimizer,
    scheduler,
    scaler,
    epoch: int,
    best_ba: float,
    best_f1: float,
    patience_counter: int,
    normalizer: dict,
    protocol_hash: str,
    registry_hash: str,
) -> dict:

    return {
        "operation_id":
            operation[
                "operation_id"
            ],

        "protocol_revision":
            "1.4",

        "protocol_hash":
            protocol_hash,

        "registry_hash":
            registry_hash,

        "seed":
            int(
                operation[
                    "seed"
                ]
            ),

        "epoch":
            int(
                epoch
            ),

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "scheduler_state_dict":
            (
                scheduler.state_dict()
                if scheduler is not None
                else None
            ),

        "scaler_state_dict":
            (
                scaler.state_dict()
                if scaler is not None
                else None
            ),

        "best_validation_balanced_accuracy":
            float(
                best_ba
            ),

        "best_validation_macro_f1":
            float(
                best_f1
            ),

        "patience_counter":
            int(
                patience_counter
            ),

        "normalizer":
            normalizer,

        "python_random_state":
            random.getstate(),

        "numpy_random_state":
            np.random.get_state(),

        "torch_rng_state":
            torch.get_rng_state(),

        "cuda_rng_state_all":
            (
                torch.cuda.get_rng_state_all()
                if torch.cuda.is_available()
                else None
            ),
    }


def restore_rng(
    checkpoint: dict,
) -> None:

    if "python_random_state" in checkpoint:

        random.setstate(
            checkpoint[
                "python_random_state"
            ]
        )

    if "numpy_random_state" in checkpoint:

        np.random.set_state(
            checkpoint[
                "numpy_random_state"
            ]
        )

    if "torch_rng_state" in checkpoint:

        torch.set_rng_state(
            checkpoint[
                "torch_rng_state"
            ]
        )

    if (
        torch.cuda.is_available()
        and checkpoint.get(
            "cuda_rng_state_all"
        )
        is not None
    ):

        torch.cuda.set_rng_state_all(
            checkpoint[
                "cuda_rng_state_all"
            ]
        )


def train_model(
    *,
    operation: dict,
    train_raw: dict[str, Any],
    validation_raw: dict[str, Any] | None,
    duration: int,
    n_classes: int,
    device: torch.device,
    protocol_hash: str,
    registry_hash: str,
    overall_position: str,
    adaptation: bool = False,
    initialization_checkpoint: Path | None = None,
) -> dict:

    op_dir = operation_directory(
        operation
    )

    op_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    completed_path = op_dir / "completed.json"
    result_path = op_dir / "result.json"
    last_path = op_dir / "last.pt"
    best_path = op_dir / "best.pt"

    if (
        completed_path.exists()
        and result_path.exists()
    ):

        completed = json.loads(
            completed_path.read_text(
                encoding="utf-8"
            )
        )

        if (
            completed.get(
                "operation_id"
            )
            == operation[
                "operation_id"
            ]
            and completed.get(
                "registry_hash"
            )
            == registry_hash
            and completed.get(
                "protocol_hash"
            )
            == protocol_hash
        ):

            return json.loads(
                result_path.read_text(
                    encoding="utf-8"
                )
            )

        raise RuntimeError(
            f"Completed checkpoint contract mismatch: {op_dir}"
        )

    model_name = str(
        operation[
            "model"
        ]
    )

    path = str(
        operation[
            "path"
        ]
    )

    seed = int(
        operation[
            "seed"
        ]
    )

    set_determinism(
        seed
    )

    if adaptation:

        if initialization_checkpoint is None:

            raise RuntimeError(
                "Adaptive training requires the selected global checkpoint."
            )

        global_checkpoint = torch.load(
            initialization_checkpoint,
            map_location="cpu",
            weights_only=False,
        )

        normalizer = global_checkpoint[
            "normalizer"
        ]

    else:

        global_checkpoint = None

        normalizer = fit_normalizer(
            train_raw,
            path,
        )

    train_data = apply_normalizer(
        train_raw,
        path,
        normalizer,
    )

    validation_data = (
        apply_normalizer(
            validation_raw,
            path,
            normalizer,
        )
        if validation_raw is not None
        else None
    )

    model = build_model(
        model_name=model_name,
        path=path,
        n_classes=n_classes,
        duration=duration,
        seed=seed,
        device=device,
    )

    if global_checkpoint is not None:

        model.load_state_dict(
            global_checkpoint[
                "model_state_dict"
            ],
            strict=True,
        )

        configure_head_only(
            model,
            model_name,
        )

    parameters = [
        parameter
        for parameter
        in model.parameters()
        if parameter.requires_grad
    ]

    if not parameters:

        raise RuntimeError(
            "No trainable parameters."
        )

    learning_rate = (
        ADAPTIVE_LR
        if adaptation
        else GLOBAL_LR
    )

    weight_decay = (
        ADAPTIVE_WEIGHT_DECAY
        if adaptation
        else GLOBAL_WEIGHT_DECAY
    )

    max_epochs = (
        ADAPTIVE_MAX_EPOCHS
        if adaptation
        else GLOBAL_MAX_EPOCHS
    )

    optimizer = torch.optim.AdamW(
        parameters,
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    scheduler = (
        None
        if adaptation
        else torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=5,
            min_lr=1e-5,
        )
    )

    try:

        scaler = torch.amp.GradScaler(
            "cuda",
            enabled=(
                device.type
                == "cuda"
            ),
        )

    except Exception:

        scaler = torch.cuda.amp.GradScaler(
            enabled=(
                device.type
                == "cuda"
            ),
        )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights(
            train_data[
                "labels"
            ],
            n_classes,
            device,
        )
    )

    sampling_weights = None

    if str(
        operation[
            "stage"
        ]
    ).startswith(
        "bbbd_"
    ):

        sampling_weights = recording_sampling_weights(
            train_data[
                "recording"
            ]
        )

    start_epoch = 1
    best_ba = float(
        "-inf"
    )
    best_f1 = float(
        "-inf"
    )
    patience_counter = 0

    if last_path.exists():

        checkpoint = torch.load(
            last_path,
            map_location=device,
            weights_only=False,
        )

        for key, expected in [
            (
                "operation_id",
                operation[
                    "operation_id"
                ],
            ),
            (
                "protocol_hash",
                protocol_hash,
            ),
            (
                "registry_hash",
                registry_hash,
            ),
        ]:

            if checkpoint.get(
                key
            ) != expected:

                raise RuntimeError(
                    f"Resume checkpoint mismatch for {key}."
                )

        model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ],
            strict=True,
        )

        optimizer.load_state_dict(
            checkpoint[
                "optimizer_state_dict"
            ]
        )

        if (
            scheduler is not None
            and checkpoint.get(
                "scheduler_state_dict"
            )
            is not None
        ):

            scheduler.load_state_dict(
                checkpoint[
                    "scheduler_state_dict"
                ]
            )

        if checkpoint.get(
            "scaler_state_dict"
        ) is not None:

            scaler.load_state_dict(
                checkpoint[
                    "scaler_state_dict"
                ]
            )

        start_epoch = (
            int(
                checkpoint[
                    "epoch"
                ]
            )
            + 1
        )

        best_ba = float(
            checkpoint[
                "best_validation_balanced_accuracy"
            ]
        )

        best_f1 = float(
            checkpoint[
                "best_validation_macro_f1"
            ]
        )

        patience_counter = int(
            checkpoint[
                "patience_counter"
            ]
        )

        normalizer = checkpoint[
            "normalizer"
        ]

        restore_rng(
            checkpoint
        )

    operation_start = time.perf_counter()

    context = (
        f"{overall_position} "
        f"{operation['stage']} "
        f"{operation['split_id']} "
        f"{model_name}/{path} "
        f"seed={seed} d={duration}s"
    )

    epoch_bar = tqdm(
        range(
            start_epoch,
            max_epochs
            + 1,
        ),
        desc=context,
        unit="epoch",
        leave=False,
        dynamic_ncols=True,
    )

    for epoch in epoch_bar:

        model.train()

        ordering = epoch_indices(
            len(
                train_data[
                    "labels"
                ]
            ),
            seed,
            epoch,
            sampling_weights,
        )

        loss_total = 0.0
        examples = 0

        for batch_start in range(
            0,
            len(
                ordering
            ),
            BATCH_SIZE,
        ):

            indices = ordering[
                batch_start:
                    batch_start
                    + BATCH_SIZE
            ]

            labels = torch.as_tensor(
                train_data[
                    "labels"
                ][
                    indices
                ],
                dtype=torch.long,
                device=device,
            )

            batch = move_batch(
                train_data,
                indices,
                path,
                device,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=(
                    device.type
                    == "cuda"
                ),
            ):

                logits = forward_model(
                    model,
                    batch,
                    model_name,
                    path,
                )

                loss = criterion(
                    logits,
                    labels,
                )

            scaler.scale(
                loss
            ).backward()

            scaler.unscale_(
                optimizer
            )

            torch.nn.utils.clip_grad_norm_(
                parameters,
                GRAD_CLIP,
            )

            scaler.step(
                optimizer
            )

            scaler.update()

            count = len(
                indices
            )

            loss_total += (
                float(
                    loss.detach().cpu()
                )
                * count
            )

            examples += count

        train_loss = (
            loss_total
            / max(
                examples,
                1,
            )
        )

        improved = False

        if adaptation:

            current_ba = float(
                "nan"
            )

            current_f1 = float(
                "nan"
            )

            # No held-out/test-based early stopping in adaptation.
            patience_counter = 0

        else:

            if validation_data is None:

                raise RuntimeError(
                    "Global training requires validation data."
                )

            validation_probabilities, _ = predict(
                model,
                validation_data,
                path,
                model_name,
                device,
            )

            validation_metrics = metric_bundle(
                validation_data[
                    "labels"
                ],
                validation_probabilities,
            )

            current_ba = validation_metrics[
                "balanced_accuracy"
            ]

            current_f1 = validation_metrics[
                "macro_f1"
            ]

            improved = (
                current_ba
                > best_ba
                + 1e-12
                or (
                    abs(
                        current_ba
                        - best_ba
                    )
                    <= 1e-12
                    and current_f1
                    > best_f1
                    + 1e-12
                )
            )

            if improved:

                best_ba = current_ba
                best_f1 = current_f1
                patience_counter = 0

            else:

                patience_counter += 1

            scheduler.step(
                current_ba
            )

        if adaptation:

            # Adaptive protocol uses all 20 calibration epochs.
            improved = (
                epoch
                == max_epochs
            )

            if improved:

                best_ba = float(
                    "nan"
                )

                best_f1 = float(
                    "nan"
                )

        checkpoint = checkpoint_payload(
            operation=operation,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            best_ba=best_ba,
            best_f1=best_f1,
            patience_counter=patience_counter,
            normalizer=normalizer,
            protocol_hash=protocol_hash,
            registry_hash=registry_hash,
        )

        atomic_torch_save(
            last_path,
            checkpoint,
        )

        if improved:

            atomic_torch_save(
                best_path,
                checkpoint,
            )

        epoch_bar.set_postfix(
            {
                "loss":
                    f"{train_loss:.4f}",

                "val":
                    (
                        "-"
                        if adaptation
                        else f"{current_ba:.4f}"
                    ),

                "best":
                    (
                        "-"
                        if adaptation
                        else f"{best_ba:.4f}"
                    ),

                "pat":
                    (
                        "-"
                        if adaptation
                        else f"{patience_counter}/{GLOBAL_PATIENCE}"
                    ),

                "lr":
                    f"{optimizer.param_groups[0]['lr']:.2e}",

                "ckpt":
                    "saved",
            },
            refresh=True,
        )

        if (
            not adaptation
            and patience_counter
            >= GLOBAL_PATIENCE
        ):

            break

    epoch_bar.close()

    if not best_path.exists():

        # Resume may begin after the final adaptive epoch.
        if adaptation and last_path.exists():

            checkpoint = torch.load(
                last_path,
                map_location="cpu",
                weights_only=False,
            )

            atomic_torch_save(
                best_path,
                checkpoint,
            )

        else:

            raise RuntimeError(
                f"No best checkpoint produced for {operation['operation_id']}."
            )

    best_checkpoint = torch.load(
        best_path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        best_checkpoint[
            "model_state_dict"
        ],
        strict=True,
    )

    validation_metrics = {}

    if validation_data is not None:

        validation_probabilities, validation_seconds = predict(
            model,
            validation_data,
            path,
            model_name,
            device,
        )

        validation_metrics = metric_bundle(
            validation_data[
                "labels"
            ],
            validation_probabilities,
        )

        validation_metrics[
            "inference_seconds"
        ] = validation_seconds

    wall_seconds = (
        time.perf_counter()
        - operation_start
    )

    result = {
        "operation_id":
            operation[
                "operation_id"
            ],

        "stage":
            operation[
                "stage"
            ],

        "protocol":
            operation[
                "protocol"
            ],

        "dataset":
            operation[
                "dataset"
            ],

        "split_id":
            operation[
                "split_id"
            ],

        "model":
            model_name,

        "path":
            path,

        "seed":
            seed,

        "duration_seconds":
            duration,

        "calibration_budget_seconds_per_class":
            (
                int(
                    float(
                        operation[
                            "calibration_budget_seconds_per_class"
                        ]
                    )
                )
                if (
                    operation.get(
                        "calibration_budget_seconds_per_class"
                    )
                    is not None
                    and not pd.isna(
                        operation.get(
                            "calibration_budget_seconds_per_class"
                        )
                    )
                )
                else None
            ),

        "adaptation":
            adaptation,

        "validation_metrics":
            validation_metrics,

        "best_checkpoint":
            str(
                best_path
            ),

        "training_wall_seconds":
            wall_seconds,

        "total_parameters":
            int(
                sum(
                    parameter.numel()
                    for parameter
                    in model.parameters()
                )
            ),

        "trainable_parameters":
            int(
                sum(
                    parameter.numel()
                    for parameter
                    in model.parameters()
                    if parameter.requires_grad
                )
            ),

        "protocol_hash":
            protocol_hash,

        "registry_hash":
            registry_hash,
    }

    atomic_json(
        result_path,
        result,
    )

    atomic_json(
        completed_path,
        {
            "operation_id":
                operation[
                    "operation_id"
                ],

            "protocol_hash":
                protocol_hash,

            "registry_hash":
                registry_hash,

            "completed":
                True,

            "completed_at_unix":
                time.time(),
        },
    )

    return result


def load_model_for_evaluation(
    *,
    result: dict,
    n_classes: int,
    device: torch.device,
) -> tuple[nn.Module, dict]:

    checkpoint_path = Path(
        result[
            "best_checkpoint"
        ]
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model = build_model(
        model_name=result[
            "model"
        ],
        path=result[
            "path"
        ],
        n_classes=n_classes,
        duration=int(
            result[
                "duration_seconds"
            ]
        ),
        seed=int(
            result[
                "seed"
            ]
        ),
        device=device,
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ],
        strict=True,
    )

    return (
        model,
        checkpoint[
            "normalizer"
        ],
    )


def save_prediction_table(
    path: Path,
    data: dict[str, Any],
    probabilities: np.ndarray,
) -> None:

    frame = pd.DataFrame(
        {
            "participant":
                data[
                    "participant"
                ].astype(
                    str
                ),

            "recording":
                data[
                    "recording"
                ].astype(
                    str
                ),

            "label":
                data[
                    "labels"
                ].astype(
                    int
                ),
        }
    )

    for class_index in range(
        probabilities.shape[
            1
        ]
    ):

        frame[
            f"p{class_index}"
        ] = probabilities[
            :,
            class_index
        ]

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    frame.to_csv(
        path,
        index=False,
        compression="gzip",
    )


def evaluate_standard_test(
    result: dict,
    test_raw: dict[str, Any],
    n_classes: int,
    device: torch.device,
    suffix: str = "test",
) -> dict:

    model, normalizer = load_model_for_evaluation(
        result=result,
        n_classes=n_classes,
        device=device,
    )

    test_data = apply_normalizer(
        test_raw,
        result[
            "path"
        ],
        normalizer,
    )

    probabilities, inference_seconds = predict(
        model,
        test_data,
        result[
            "path"
        ],
        result[
            "model"
        ],
        device,
    )

    metrics = metric_bundle(
        test_data[
            "labels"
        ],
        probabilities,
    )

    metrics[
        "inference_seconds"
    ] = inference_seconds

    metrics[
        "inference_ms_per_window"
    ] = (
        1000.0
        * inference_seconds
        / len(
            test_data[
                "labels"
            ]
        )
    )

    op_dir = operation_directory(
        result
    )

    save_prediction_table(
        op_dir
        / f"{suffix}_predictions.csv.gz",
        test_data,
        probabilities,
    )

    atomic_json(
        op_dir
        / f"{suffix}_metrics.json",
        metrics,
    )

    return metrics


def evaluate_bbbd_test(
    result: dict,
    validation_raw: dict[str, Any],
    test_raw: dict[str, Any],
    device: torch.device,
) -> dict:

    model, normalizer = load_model_for_evaluation(
        result=result,
        n_classes=2,
        device=device,
    )

    validation_data = apply_normalizer(
        validation_raw,
        result[
            "path"
        ],
        normalizer,
    )

    test_data = apply_normalizer(
        test_raw,
        result[
            "path"
        ],
        normalizer,
    )

    validation_probabilities, _ = predict(
        model,
        validation_data,
        result[
            "path"
        ],
        result[
            "model"
        ],
        device,
    )

    validation_recordings = aggregate_recordings(
        validation_data,
        validation_probabilities,
    )

    threshold, validation_metrics = choose_threshold(
        validation_recordings
    )

    test_probabilities, inference_seconds = predict(
        model,
        test_data,
        result[
            "path"
        ],
        result[
            "model"
        ],
        device,
    )

    test_recordings = aggregate_recordings(
        test_data,
        test_probabilities,
    )

    test_metrics = binary_metrics_at_threshold(
        test_recordings[
            "label"
        ].to_numpy(
            dtype=int
        ),
        test_recordings[
            "p1"
        ].to_numpy(
            dtype=float
        ),
        threshold,
    )

    test_metrics[
        "threshold"
    ] = threshold

    test_metrics[
        "validation_balanced_accuracy_at_threshold"
    ] = validation_metrics[
        "balanced_accuracy"
    ]

    test_metrics[
        "validation_macro_f1_at_threshold"
    ] = validation_metrics[
        "macro_f1"
    ]

    test_metrics[
        "window_inference_seconds"
    ] = inference_seconds

    test_metrics[
        "window_inference_ms_per_window"
    ] = (
        1000.0
        * inference_seconds
        / len(
            test_data[
                "labels"
            ]
        )
    )

    participant_metrics = []

    for participant, group in test_recordings.groupby(
        "participant"
    ):

        labels = group[
            "label"
        ].to_numpy(
            dtype=int
        )

        positive = group[
            "p1"
        ].to_numpy(
            dtype=float
        )

        if len(
            np.unique(
                labels
            )
        ) < 2:

            continue

        metrics = binary_metrics_at_threshold(
            labels,
            positive,
            threshold,
        )

        participant_metrics.append(
            metrics
        )

    if participant_metrics:

        for metric_name in [
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "roc_auc",
            "pr_auc",
        ]:

            test_metrics[
                f"participant_macro_{metric_name}"
            ] = float(
                np.mean(
                    [
                        item[
                            metric_name
                        ]
                        for item
                        in participant_metrics
                    ]
                )
            )

    op_dir = operation_directory(
        result
    )

    op_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    test_recordings.to_csv(
        op_dir
        / "test_recording_predictions.csv",
        index=False,
    )

    validation_recordings.to_csv(
        op_dir
        / "validation_recording_predictions.csv",
        index=False,
    )

    atomic_json(
        op_dir
        / "test_metrics.json",
        test_metrics,
    )

    return test_metrics


def completed_operation_count(
    operations: pd.DataFrame,
    protocol_hash: str,
    registry_hash: str,
) -> int:

    count = 0

    for row in operations.to_dict(
        orient="records"
    ):

        completed = (
            operation_directory(
                row
            )
            / "completed.json"
        )

        if not completed.exists():
            continue

        try:

            payload = json.loads(
                completed.read_text(
                    encoding="utf-8"
                )
            )

            if (
                payload.get(
                    "operation_id"
                )
                == row[
                    "operation_id"
                ]
                and payload.get(
                    "protocol_hash"
                )
                == protocol_hash
                and payload.get(
                    "registry_hash"
                )
                == registry_hash
            ):

                count += 1

        except Exception:
            continue

    return count



def operation_completed_before_start(
    operation: dict,
    protocol_hash: str,
    registry_hash: str,
) -> bool:

    op_dir = operation_directory(
        operation
    )

    completed_path = (
        op_dir
        / "completed.json"
    )

    result_path = (
        op_dir
        / "result.json"
    )

    if (
        not completed_path.exists()
        or not result_path.exists()
    ):

        return False

    try:

        payload = json.loads(
            completed_path.read_text(
                encoding="utf-8"
            )
        )

    except Exception:

        return False

    return (
        payload.get(
            "operation_id"
        )
        == operation[
            "operation_id"
        ]
        and payload.get(
            "protocol_hash"
        )
        == protocol_hash
        and payload.get(
            "registry_hash"
        )
        == registry_hash
        and bool(
            payload.get(
                "completed",
                False,
            )
        )
    )


def flatten_result_row(
    result: dict,
    test_metrics: dict | None = None,
) -> dict:

    validation = result.get(
        "validation_metrics",
        {}
    )

    test_metrics = (
        test_metrics
        or {}
    )

    return {
        "operation_id":
            result[
                "operation_id"
            ],

        "stage":
            result[
                "stage"
            ],

        "protocol":
            result[
                "protocol"
            ],

        "dataset":
            result[
                "dataset"
            ],

        "split_id":
            result[
                "split_id"
            ],

        "model":
            result[
                "model"
            ],

        "path":
            result[
                "path"
            ],

        "seed":
            result[
                "seed"
            ],

        "duration_seconds":
            result[
                "duration_seconds"
            ],

        "calibration_budget_seconds_per_class":
            result.get(
                "calibration_budget_seconds_per_class"
            ),

        "validation_balanced_accuracy":
            validation.get(
                "balanced_accuracy"
            ),

        "validation_macro_f1":
            validation.get(
                "macro_f1"
            ),

        "test_accuracy":
            test_metrics.get(
                "accuracy"
            ),

        "test_balanced_accuracy":
            test_metrics.get(
                "balanced_accuracy"
            ),

        "test_macro_f1":
            test_metrics.get(
                "macro_f1"
            ),

        "test_roc_auc":
            test_metrics.get(
                "roc_auc"
            ),

        "test_pr_auc":
            test_metrics.get(
                "pr_auc"
            ),

        "threshold":
            test_metrics.get(
                "threshold"
            ),

        "participant_macro_balanced_accuracy":
            test_metrics.get(
                "participant_macro_balanced_accuracy"
            ),

        "participant_macro_macro_f1":
            test_metrics.get(
                "participant_macro_macro_f1"
            ),

        "training_wall_seconds":
            result[
                "training_wall_seconds"
            ],

        "total_parameters":
            result[
                "total_parameters"
            ],

        "trainable_parameters":
            result[
                "trainable_parameters"
            ],
    }


def write_results_table(
    rows: list[dict],
) -> None:

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    pd.DataFrame(
        rows
    ).to_csv(
        RESULTS_CSV,
        index=False,
    )


def run_preflight() -> None:

    global INTERNAL_CACHE_MAP
    global BBBD_FILE_META

    print(
        "===== REAL TRAINING LAUNCH PREFLIGHT ====="
    )

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA is required for the real deep-learning run."
        )

    registry_hash = sha256_file(
        REGISTRY
    )

    if registry_hash != EXPECTED_REGISTRY_SHA256:

        raise RuntimeError(
            f"Registry v3 SHA mismatch: {registry_hash}"
        )

    registry = pd.read_csv(
        REGISTRY,
        low_memory=False,
    )

    operations = expand_execution_registry(
        registry
    )

    if len(
        operations
    ) != EXPECTED_OPERATIONS:

        raise RuntimeError(
            f"Expected 3,000 operations; found {len(operations)}."
        )

    if operations[
        "operation_id"
    ].nunique() != EXPECTED_OPERATIONS:

        raise RuntimeError(
            "Operation IDs are not globally unique."
        )

    protocol = yaml.safe_load(
        CONFIG.read_text(
            encoding="utf-8"
        )
    )

    if str(
        protocol[
            "protocol_revision"
        ]
    ) != "1.4":

        raise RuntimeError(
            "Protocol revision is not 1.4."
        )

    INTERNAL_CACHE_MAP = discover_internal_caches()

    for duration, expected_rows in INTERNAL_EXPECTED_COUNTS.items():

        data = load_internal_cache(
            duration
        )

        if len(
            data[
                "labels"
            ]
        ) != expected_rows:

            raise RuntimeError(
                f"Internal {duration}s row-count mismatch."
            )

    BBBD_FILE_META = discover_bbbd_files(
        registry
    )

    device = torch.device(
        "cuda"
    )

    # Model-interface checks: initialization + forward only.
    representative_checks = [
        (
            "ShallowConvNet",
            "EEG",
            4,
            3,
        ),
        (
            "MultibranchTCN",
            "EEG",
            32,
            3,
        ),
        (
            "MultibranchTCN",
            "ECG",
            32,
            3,
        ),
        (
            "MultibranchTCN",
            "ECG_EEG",
            32,
            3,
        ),
        (
            "MultibranchTCN",
            "Pupil",
            8,
            3,
        ),
        (
            "MultibranchTCN",
            "ECG_Pupil",
            8,
            3,
        ),
        (
            "MultibranchTCN",
            "EEG_Pupil",
            8,
            3,
        ),
        (
            "MultibranchTCN",
            "ECG_EEG_Pupil",
            8,
            3,
        ),
    ]

    for model_name, path, duration, n_classes in representative_checks:

        source = load_internal_cache(
            duration
        )

        sample = subset_data(
            source,
            np.asarray(
                [
                    0,
                    1,
                ]
            ),
            path,
        )

        normalizer = fit_normalizer(
            sample,
            path,
        )

        sample = apply_normalizer(
            sample,
            path,
            normalizer,
        )

        model = build_model(
            model_name=model_name,
            path=path,
            n_classes=n_classes,
            duration=duration,
            seed=42,
            device=device,
        )

        with torch.no_grad():

            logits = forward_model(
                model,
                move_batch(
                    sample,
                    np.asarray(
                        [
                            0,
                            1,
                        ]
                    ),
                    path,
                    device,
                ),
                model_name,
                path,
            )

        if logits.shape != (
            2,
            n_classes,
        ):

            raise RuntimeError(
                f"Forward preflight failed for {model_name}/{path}: "
                f"{tuple(logits.shape)}"
            )

        del model

        torch.cuda.empty_cache()

    # Exact binding helpers must still function.
    index_map, replay = regenerate_frozen_conventional_index_map()

    if len(
        replay
    ) != 3000:

        raise RuntimeError(
            "Historical conventional replay failed."
        )

    conventional_row = (
        registry.loc[
            registry[
                "stage"
            ]
            == "internal_conventional"
        ]
        .iloc[
            0
        ]
        .to_dict()
    )

    bind_internal_conventional_row(
        conventional_row,
        index_map,
    )

    nested_operation = (
        operations.loc[
            operations[
                "stage"
            ]
            == "internal_nested_loso"
        ]
        .iloc[
            0
        ]
        .to_dict()
    )

    bind_internal_participant_split(
        nested_operation,
        duration_seconds=int(
            nested_operation[
                "duration_candidate_seconds"
            ]
        ),
    )

    adaptive_operation = (
        operations.loc[
            operations[
                "stage"
            ]
            == "internal_subject_adaptive"
        ]
        .iloc[
            0
        ]
        .to_dict()
    )

    adaptive_temporal_roles(
        row=adaptive_operation,
        duration_seconds=8,
    )

    print(
        "CUDA device:",
        torch.cuda.get_device_name(
            0
        ),
    )

    print(
        "Registry v3 SHA256:",
        registry_hash,
    )

    print(
        "Operations:",
        len(
            operations
        ),
    )

    print(
        "Internal caches:",
        {
            duration:
                str(
                    path
                )
            for duration, path
            in INTERNAL_CACHE_MAP.items()
        },
    )

    print(
        "BBBD recording caches:",
        len(
            BBBD_FILE_META
        ),
    )

    print(
        "BBBD participants:",
        BBBD_FILE_META[
            "participant"
        ].nunique(),
    )

    print(
        "All model/path forward checks: PASSED"
    )

    print(
        "Optimization performed: False"
    )

    print(
        "Performance observed: False"
    )

    print(
        "REAL TRAINING PREFLIGHT: PASSED"
    )


def run_real_training() -> None:

    global INTERNAL_CACHE_MAP
    global BBBD_FILE_META

    if not AUTHORIZATION.is_file():

        raise RuntimeError(
            "Real-training authorization artifact is missing."
        )

    authorization = json.loads(
        AUTHORIZATION.read_text(
            encoding="utf-8"
        )
    )

    if not authorization.get(
        "authorized",
        False,
    ):

        raise RuntimeError(
            "Real-training authorization is not active."
        )

    if authorization.get(
        "pretraining_commit"
    ) != "f9dd38b":

        raise RuntimeError(
            "Authorization is not bound to frozen pretraining commit f9dd38b."
        )

    if sha256_file(
        REGISTRY
    ) != EXPECTED_REGISTRY_SHA256:

        raise RuntimeError(
            "Registry v3 changed after authorization."
        )

    protocol_hash = sha256_file(
        CONFIG
    )

    registry_hash = sha256_file(
        REGISTRY
    )

    registry = pd.read_csv(
        REGISTRY,
        low_memory=False,
    )

    operations = expand_execution_registry(
        registry
    )

    if len(
        operations
    ) != EXPECTED_OPERATIONS:

        raise RuntimeError(
            "Active execution plan is not exactly 3,000 operations."
        )

    INTERNAL_CACHE_MAP = discover_internal_caches()

    BBBD_FILE_META = discover_bbbd_files(
        registry
    )

    device = torch.device(
        "cuda"
    )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    index_map, _ = regenerate_frozen_conventional_index_map()

    already_completed = completed_operation_count(
        operations,
        protocol_hash,
        registry_hash,
    )

    overall = tqdm(
        total=EXPECTED_OPERATIONS,
        initial=already_completed,
        desc="Deep v3 operations",
        unit="op",
        dynamic_ncols=True,
    )

    wall_times: list[float] = []

    result_rows: list[dict] = []

    nested_selection_rows: list[dict] = []

    nested_selection_map: dict[
        tuple,
        dict,
    ] = {}

    def update_overall(
        elapsed_seconds: float,
        operation: dict,
    ):

        wall_times.append(
            elapsed_seconds
        )

        mean_seconds = float(
            np.mean(
                wall_times
            )
        )

        remaining = (
            EXPECTED_OPERATIONS
            - overall.n
            - 1
        )

        eta_seconds = max(
            remaining,
            0,
        ) * mean_seconds

        overall.set_description(
            (
                f"{operation['stage']} "
                f"{operation['split_id']} "
                f"{operation['model']}/{operation['path']} "
                f"seed={operation['seed']}"
            )
        )

        overall.set_postfix(
            {
                "avg_op":
                    f"{mean_seconds:.1f}s",

                "ETA":
                    (
                        f"{eta_seconds / 3600.0:.2f}h"
                    ),
            },
            refresh=False,
        )

        overall.update(
            1
        )

    # ============================================================
    # 1. INTERNAL CONVENTIONAL
    # ============================================================

    conventional = operations.loc[
        operations[
            "stage"
        ]
        == "internal_conventional"
    ].copy()

    for operation in conventional.to_dict(
        orient="records"
    ):

        started = time.perf_counter()

        completed_before = operation_completed_before_start(
            operation,
            protocol_hash,
            registry_hash,
        )

        binding = bind_internal_conventional_row(
            operation,
            index_map,
        )

        path = operation[
            "path"
        ]

        train = internal_partition_data(
            4,
            binding[
                "train_indices"
            ],
            path,
        )

        validation = internal_partition_data(
            4,
            binding[
                "validation_indices"
            ],
            path,
        )

        test = internal_partition_data(
            4,
            binding[
                "test_indices"
            ],
            path,
        )

        result = train_model(
            operation=operation,
            train_raw=train,
            validation_raw=validation,
            duration=4,
            n_classes=3,
            device=device,
            protocol_hash=protocol_hash,
            registry_hash=registry_hash,
            overall_position=(
                f"job {overall.n + 1}/{EXPECTED_OPERATIONS}"
            ),
        )

        test_metrics = evaluate_standard_test(
            result,
            test,
            3,
            device,
        )

        result_rows.append(
            flatten_result_row(
                result,
                test_metrics,
            )
        )

        write_results_table(
            result_rows
        )

        if not completed_before:

            update_overall(
                time.perf_counter()
                - started,
                operation,
            )

    # ============================================================
    # 2. STRICT NESTED LOSO
    # ============================================================

    nested = operations.loc[
        operations[
            "stage"
        ]
        == "internal_nested_loso"
    ].copy()

    grouping = [
        "split_id",
        "model",
        "path",
        "seed",
    ]

    for key, group in nested.groupby(
        grouping,
        sort=True,
    ):

        candidate_results = []

        candidate_operations = []

        for operation in (
            group.sort_values(
                "duration_candidate_seconds"
            )
            .to_dict(
                orient="records"
            )
        ):

            started = time.perf_counter()

            completed_before = operation_completed_before_start(
                operation,
                protocol_hash,
                registry_hash,
            )

            duration = int(
                operation[
                    "duration_candidate_seconds"
                ]
            )

            binding = bind_internal_participant_split(
                operation,
                duration_seconds=duration,
            )

            path = operation[
                "path"
            ]

            train = internal_partition_data(
                duration,
                binding[
                    "train_indices"
                ],
                path,
            )

            validation = internal_partition_data(
                duration,
                binding[
                    "validation_indices"
                ],
                path,
            )

            result = train_model(
                operation=operation,
                train_raw=train,
                validation_raw=validation,
                duration=duration,
                n_classes=3,
                device=device,
                protocol_hash=protocol_hash,
                registry_hash=registry_hash,
                overall_position=(
                    f"job {overall.n + 1}/{EXPECTED_OPERATIONS}"
                ),
            )

            candidate_results.append(
                result
            )

            candidate_operations.append(
                (
                    operation,
                    binding,
                )
            )

            result_rows.append(
                flatten_result_row(
                    result,
                    None,
                )
            )

            write_results_table(
                result_rows
            )

            if not completed_before:

                update_overall(
                    time.perf_counter()
                    - started,
                    operation,
                )

        ranked = sorted(
            candidate_results,
            key=lambda item:
                (
                    item[
                        "validation_metrics"
                    ][
                        "balanced_accuracy"
                    ],
                    item[
                        "validation_metrics"
                    ][
                        "macro_f1"
                    ],
                    -int(
                        item[
                            "duration_seconds"
                        ]
                    ),
                ),
            reverse=True,
        )

        selected = ranked[
            0
        ]

        selected_duration = int(
            selected[
                "duration_seconds"
            ]
        )

        selected_operation = None
        selected_binding = None

        for operation, binding in candidate_operations:

            if int(
                operation[
                    "duration_candidate_seconds"
                ]
            ) == selected_duration:

                selected_operation = operation
                selected_binding = binding
                break

        if selected_binding is None:

            raise RuntimeError(
                "Could not recover selected nested binding."
            )

        test = internal_partition_data(
            selected_duration,
            selected_binding[
                "test_indices"
            ],
            selected[
                "path"
            ],
        )

        selected_test_metrics = evaluate_standard_test(
            selected,
            test,
            3,
            device,
            suffix="selected_test",
        )

        selection_key = (
            selected[
                "split_id"
            ],
            selected[
                "model"
            ],
            selected[
                "path"
            ],
            int(
                selected[
                    "seed"
                ]
            ),
        )

        selection_payload = {
            "split_id":
                selected[
                    "split_id"
                ],

            "model":
                selected[
                    "model"
                ],

            "path":
                selected[
                    "path"
                ],

            "seed":
                int(
                    selected[
                        "seed"
                    ]
                ),

            "selected_duration_seconds":
                selected_duration,

            "validation_balanced_accuracy":
                selected[
                    "validation_metrics"
                ][
                    "balanced_accuracy"
                ],

            "validation_macro_f1":
                selected[
                    "validation_metrics"
                ][
                    "macro_f1"
                ],

            "selected_global_checkpoint":
                selected[
                    "best_checkpoint"
                ],

            "test_accuracy":
                selected_test_metrics[
                    "accuracy"
                ],

            "test_balanced_accuracy":
                selected_test_metrics[
                    "balanced_accuracy"
                ],

            "test_macro_f1":
                selected_test_metrics[
                    "macro_f1"
                ],

            "test_roc_auc":
                selected_test_metrics[
                    "roc_auc"
                ],

            "test_pr_auc":
                selected_test_metrics[
                    "pr_auc"
                ],
        }

        nested_selection_map[
            selection_key
        ] = selection_payload

        nested_selection_rows.append(
            selection_payload
        )

        pd.DataFrame(
            nested_selection_rows
        ).to_csv(
            NESTED_SELECTION_CSV,
            index=False,
        )

    # ============================================================
    # 3. SUBJECT-ADAPTIVE
    # ============================================================

    adaptive = operations.loc[
        operations[
            "stage"
        ]
        == "internal_subject_adaptive"
    ].copy()

    for operation in adaptive.to_dict(
        orient="records"
    ):

        started = time.perf_counter()

        completed_before = operation_completed_before_start(
            operation,
            protocol_hash,
            registry_hash,
        )

        key = (
            operation[
                "split_id"
            ],
            operation[
                "model"
            ],
            operation[
                "path"
            ],
            int(
                operation[
                    "seed"
                ]
            ),
        )

        if key not in nested_selection_map:

            raise RuntimeError(
                f"Missing nested selection for adaptive operation: {key}"
            )

        selected = nested_selection_map[
            key
        ]

        duration = int(
            selected[
                "selected_duration_seconds"
            ]
        )

        roles = adaptive_temporal_roles(
            row=operation,
            duration_seconds=duration,
        )

        path = operation[
            "path"
        ]

        calibration = internal_partition_data(
            duration,
            roles[
                "calibration_indices"
            ],
            path,
        )

        test = internal_partition_data(
            duration,
            roles[
                "test_indices"
            ],
            path,
        )

        result = train_model(
            operation=operation,
            train_raw=calibration,
            validation_raw=None,
            duration=duration,
            n_classes=3,
            device=device,
            protocol_hash=protocol_hash,
            registry_hash=registry_hash,
            overall_position=(
                f"job {overall.n + 1}/{EXPECTED_OPERATIONS}"
            ),
            adaptation=True,
            initialization_checkpoint=Path(
                selected[
                    "selected_global_checkpoint"
                ]
            ),
        )

        test_metrics = evaluate_standard_test(
            result,
            test,
            3,
            device,
        )

        result_rows.append(
            flatten_result_row(
                result,
                test_metrics,
            )
        )

        write_results_table(
            result_rows
        )

        if not completed_before:

            update_overall(
                time.perf_counter()
                - started,
                operation,
            )

    # ============================================================
    # 4. BBBD WITHIN + CROSS EXPERIMENT
    # ============================================================

    for stage in [
        "bbbd_within_experiment",
        "bbbd_cross_experiment",
    ]:

        stage_operations = operations.loc[
            operations[
                "stage"
            ]
            == stage
        ].copy()

        for operation in stage_operations.to_dict(
            orient="records"
        ):

            started = time.perf_counter()

            completed_before = operation_completed_before_start(
                operation,
                protocol_hash,
                registry_hash,
            )

            path = operation[
                "path"
            ]

            train = load_bbbd_partition(
                parse_people(
                    operation[
                        "train_participants"
                    ]
                ),
                path,
            )

            validation = load_bbbd_partition(
                parse_people(
                    operation[
                        "validation_participants"
                    ]
                ),
                path,
            )

            test = load_bbbd_partition(
                parse_people(
                    operation[
                        "test_participants"
                    ]
                ),
                path,
            )

            result = train_model(
                operation=operation,
                train_raw=train,
                validation_raw=validation,
                duration=4,
                n_classes=2,
                device=device,
                protocol_hash=protocol_hash,
                registry_hash=registry_hash,
                overall_position=(
                    f"job {overall.n + 1}/{EXPECTED_OPERATIONS}"
                ),
            )

            test_metrics = evaluate_bbbd_test(
                result,
                validation,
                test,
                device,
            )

            result_rows.append(
                flatten_result_row(
                    result,
                    test_metrics,
                )
            )

            write_results_table(
                result_rows
            )

            if not completed_before:

                update_overall(
                    time.perf_counter()
                    - started,
                    operation,
                )

    overall.close()

    if completed_operation_count(
        operations,
        protocol_hash,
        registry_hash,
    ) != EXPECTED_OPERATIONS:

        raise RuntimeError(
            "Run ended without 3,000 completed operation checkpoints."
        )

    print(
        "\n===== REAL DEEP-LEARNING RUN COMPLETE ====="
    )

    print(
        "Completed operations: 3,000 / 3,000"
    )

    print(
        "Results:",
        RESULTS_CSV,
    )

    print(
        "Nested selections:",
        NESTED_SELECTION_CSV,
    )

    print(
        "Checkpoint root:",
        OUTPUT_ROOT
        / "operations",
    )


def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--preflight",
        action="store_true",
    )

    parser.add_argument(
        "--run",
        action="store_true",
    )

    args = parser.parse_args()

    if args.preflight == args.run:

        raise RuntimeError(
            "Choose exactly one of --preflight or --run."
        )

    if args.preflight:

        run_preflight()

    else:

        run_real_training()


if __name__ == "__main__":
    main()
