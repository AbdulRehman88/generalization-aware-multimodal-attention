
"""Frozen deterministic training runtime for deep-learning baselines.

This module implements the execution mechanics specified by protocol
revision 1.3.

It does not itself select or launch real research jobs. Real-data
execution remains blocked by the protocol execution gate until the
registry/data-binding preflight is completed.

Implemented here:
- deterministic seed control;
- per-channel training-only z-score normalization;
- inverse-frequency class weighting;
- CUDA-only FP16 AMP;
- validation-only early stopping;
- ReduceLROnPlateau scheduling;
- atomic epoch checkpoints;
- exact checkpoint/resume state restoration;
- final-head-only subject adaptation;
- registry expansion to fit/adaptation operations;
- progress bars with elapsed time and ETA;
- standard classification metrics.

No hyperparameter search is implemented.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader, Dataset

from src.models.deep_learning_baselines import (
    LOCKED_MODALITY_PATHS,
    MultibranchTCN,
    ShallowConvNet,
)


ROOT = Path(__file__).resolve().parents[2]

REGISTRY_V2 = (
    ROOT
    / "artifacts/revision/manifests/"
      "deep_learning_execution_registry_v2.csv"
)


REGISTRY_V3 = (
    ROOT
    / "artifacts/revision/manifests/"
      "deep_learning_execution_registry_v3.csv"
)

PROTOCOL_PATH = (
    ROOT
    / "configs/deep_learning_baselines.yaml"
)


MODALITY_NAMES = (
    "EEG",
    "ECG",
    "Pupil",
)


# =====================================================================
# HASHING / SERIALIZATION
# =====================================================================

def canonical_json_hash(
    payload,
) -> str:

    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(
            ",",
            ":",
        ),
        default=str,
    ).encode(
        "utf-8"
    )

    return hashlib.sha256(
        encoded
    ).hexdigest()


def file_sha256(
    path: Path,
) -> str:

    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as stream:

        for block in iter(
            lambda:
                stream.read(
                    1024 * 1024
                ),
            b"",
        ):
            digest.update(
                block
            )

    return digest.hexdigest()


def atomic_json_write(
    path: Path,
    payload: dict,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_name(
        path.name
        + ".tmp"
    )

    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )

    os.replace(
        temporary,
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

    temporary = path.with_name(
        path.name
        + ".tmp"
    )

    torch.save(
        payload,
        temporary,
    )

    os.replace(
        temporary,
        path,
    )


# =====================================================================
# DETERMINISM
# =====================================================================

def set_global_determinism(
    seed: int,
) -> None:

    seed = int(
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
        True
    )

    if hasattr(
        torch.backends,
        "cudnn",
    ):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def capture_rng_state() -> dict:

    cuda_states = None

    if torch.cuda.is_available():
        cuda_states = [
            state.cpu()
            for state
            in torch.cuda.get_rng_state_all()
        ]

    return {
        "python_random_state":
            random.getstate(),

        "numpy_random_state":
            np.random.get_state(),

        "torch_cpu_rng_state":
            torch.get_rng_state().cpu(),

        "torch_cuda_rng_state_all_devices_when_cuda":
            cuda_states,
    }


def restore_rng_state(
    state: Mapping,
) -> None:

    random.setstate(
        state[
            "python_random_state"
        ]
    )

    np.random.set_state(
        state[
            "numpy_random_state"
        ]
    )

    torch.set_rng_state(
        state[
            "torch_cpu_rng_state"
        ].cpu()
    )

    cuda_states = state.get(
        "torch_cuda_rng_state_all_devices_when_cuda"
    )

    if (
        cuda_states is not None
        and torch.cuda.is_available()
    ):
        torch.cuda.set_rng_state_all(
            [
                value.to(
                    "cpu"
                )
                for value
                in cuda_states
            ]
        )


def dataloader_seed(
    training_seed: int,
    epoch: int,
) -> int:

    return int(
        training_seed
    ) + (
        int(
            epoch
        )
        * 100_003
    )


# =====================================================================
# NORMALIZATION
# =====================================================================

@dataclass
class ChannelStatistics:
    mean: np.ndarray
    std: np.ndarray


class ChannelZScoreNormalizer:
    """Per-channel z-score fit using training tensors only."""

    def __init__(
        self,
        *,
        epsilon: float = 1.0e-12,
    ) -> None:

        self.epsilon = float(
            epsilon
        )

        self.statistics: dict[
            str,
            ChannelStatistics,
        ] = {}

        self.fitted = False


    def fit(
        self,
        arrays: Mapping[
            str,
            np.ndarray,
        ],
    ) -> "ChannelZScoreNormalizer":

        if self.fitted:
            raise RuntimeError(
                "Normalizer is already fitted."
            )

        if not arrays:
            raise ValueError(
                "No training arrays supplied."
            )

        for modality, values in arrays.items():

            array = np.asarray(
                values
            )

            if array.ndim != 3:
                raise ValueError(
                    f"{modality}: expected [N,C,T], "
                    f"received {array.shape}."
                )

            if array.shape[
                0
            ] == 0:
                raise ValueError(
                    f"{modality}: empty training array."
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
                np.isfinite(
                    std
                )
                & (
                    std
                    > self.epsilon
                ),
                std,
                1.0,
            )

            if not np.isfinite(
                mean
            ).all():
                raise RuntimeError(
                    f"{modality}: non-finite training mean."
                )

            if not np.isfinite(
                std
            ).all():
                raise RuntimeError(
                    f"{modality}: non-finite training std."
                )

            self.statistics[
                str(
                    modality
                )
            ] = ChannelStatistics(
                mean=mean.astype(
                    np.float64,
                    copy=False,
                ),
                std=std.astype(
                    np.float64,
                    copy=False,
                ),
            )

        self.fitted = True

        return self


    def transform(
        self,
        arrays: Mapping[
            str,
            np.ndarray,
        ],
    ) -> dict[
        str,
        np.ndarray,
    ]:

        if not self.fitted:
            raise RuntimeError(
                "Normalizer has not been fitted."
            )

        transformed = {}

        for modality, values in arrays.items():

            if modality not in self.statistics:
                raise KeyError(
                    f"No training statistics for modality {modality}."
                )

            array = np.asarray(
                values,
                dtype=np.float32,
            )

            stats = self.statistics[
                modality
            ]

            mean = stats.mean[
                None,
                :,
                None,
            ]

            std = stats.std[
                None,
                :,
                None,
            ]

            output = (
                (
                    array.astype(
                        np.float64,
                        copy=False,
                    )
                    - mean
                )
                / std
            ).astype(
                np.float32
            )

            if not np.isfinite(
                output
            ).all():
                raise RuntimeError(
                    f"{modality}: non-finite normalized values."
                )

            transformed[
                modality
            ] = output

        return transformed


    def state_dict(
        self,
    ) -> dict:

        if not self.fitted:
            raise RuntimeError(
                "Cannot serialize an unfitted normalizer."
            )

        return {
            "epsilon":
                self.epsilon,

            "statistics":
                {
                    modality:
                        {
                            "mean":
                                stats.mean.tolist(),

                            "std":
                                stats.std.tolist(),
                        }
                    for modality, stats
                    in self.statistics.items()
                },
        }


    @classmethod
    def from_state_dict(
        cls,
        payload: Mapping,
    ) -> "ChannelZScoreNormalizer":

        instance = cls(
            epsilon=float(
                payload[
                    "epsilon"
                ]
            )
        )

        for modality, values in payload[
            "statistics"
        ].items():

            instance.statistics[
                modality
            ] = ChannelStatistics(
                mean=np.asarray(
                    values[
                        "mean"
                    ],
                    dtype=np.float64,
                ),
                std=np.asarray(
                    values[
                        "std"
                    ],
                    dtype=np.float64,
                ),
            )

        instance.fitted = True

        return instance


# =====================================================================
# CLASS / SAMPLE WEIGHTING
# =====================================================================

def inverse_frequency_class_weights(
    labels: np.ndarray,
    *,
    n_classes: int,
) -> np.ndarray:

    labels = np.asarray(
        labels,
        dtype=np.int64,
    )

    counts = np.bincount(
        labels,
        minlength=int(
            n_classes
        ),
    ).astype(
        np.float64
    )

    if np.any(
        counts <= 0
    ):
        raise ValueError(
            f"Training partition is missing class(es): counts={counts.tolist()}"
        )

    total = float(
        counts.sum()
    )

    weights = (
        total
        / (
            float(
                n_classes
            )
            * counts
        )
    )

    return weights.astype(
        np.float32
    )


# =====================================================================
# MODEL CREATION / FINAL-HEAD ADAPTATION
# =====================================================================

def build_frozen_model(
    *,
    model_name: str,
    path: str,
    n_classes: int,
    n_times: int,
) -> torch.nn.Module:

    if model_name == "ShallowConvNet":

        if path != "EEG":
            raise ValueError(
                "ShallowConvNet is EEG-only."
            )

        return ShallowConvNet(
            n_classes=int(
                n_classes
            ),
            n_times=int(
                n_times
            ),
        )

    if model_name == "MultibranchTCN":

        return MultibranchTCN(
            path=str(
                path
            ),
            n_classes=int(
                n_classes
            ),
        )

    raise ValueError(
        f"Unknown model: {model_name}"
    )


def build_seeded_frozen_model(
    *,
    training_seed: int,
    model_name: str,
    path: str,
    n_classes: int,
    n_times: int,
) -> torch.nn.Module:
    """Apply the frozen seed before neural-network initialization."""

    set_global_determinism(
        int(
            training_seed
        )
    )

    return build_frozen_model(
        model_name=model_name,
        path=path,
        n_classes=n_classes,
        n_times=n_times,
    )


def configure_final_head_only(
    model: torch.nn.Module,
) -> list[str]:

    for parameter in model.parameters():
        parameter.requires_grad = False

    if isinstance(
        model,
        ShallowConvNet,
    ):

        module = model.classifier
        expected_prefix = "classifier."

    elif isinstance(
        model,
        MultibranchTCN,
    ):

        module = model.fusion[
            3
        ]
        expected_prefix = "fusion.3."

    else:
        raise TypeError(
            f"Unsupported model type: {type(model)}"
        )

    for parameter in module.parameters():
        parameter.requires_grad = True

    names = [
        name
        for name, parameter
        in model.named_parameters()
        if parameter.requires_grad
    ]

    if not names:
        raise RuntimeError(
            "Final-head adaptation exposed no trainable parameters."
        )

    if not all(
        name.startswith(
            expected_prefix
        )
        for name
        in names
    ):
        raise RuntimeError(
            f"Unexpected trainable parameter names: {names}"
        )

    return names


# =====================================================================
# DATASET / LOADER
# =====================================================================

class ArrayClassificationDataset(
    Dataset
):

    def __init__(
        self,
        *,
        arrays: Mapping[
            str,
            np.ndarray,
        ],
        labels: np.ndarray,
        sample_weights: np.ndarray | None = None,
    ) -> None:

        if not arrays:
            raise ValueError(
                "Dataset requires at least one modality."
            )

        labels = np.asarray(
            labels,
            dtype=np.int64,
        )

        n = int(
            labels.shape[
                0
            ]
        )

        self.arrays = {}

        for modality, values in arrays.items():

            array = np.asarray(
                values,
                dtype=np.float32,
            )

            if array.ndim != 3:
                raise ValueError(
                    f"{modality}: expected [N,C,T]."
                )

            if array.shape[
                0
            ] != n:
                raise ValueError(
                    f"{modality}: label length mismatch."
                )

            self.arrays[
                modality
            ] = array

        self.labels = labels

        if sample_weights is None:

            self.sample_weights = np.ones(
                n,
                dtype=np.float32,
            )

        else:

            sample_weights = np.asarray(
                sample_weights,
                dtype=np.float32,
            )

            if sample_weights.shape != (
                n,
            ):
                raise ValueError(
                    "Sample-weight length mismatch."
                )

            if not np.isfinite(
                sample_weights
            ).all():
                raise ValueError(
                    "Sample weights must be finite."
                )

            if np.any(
                sample_weights <= 0
            ):
                raise ValueError(
                    "Sample weights must be positive."
                )

            self.sample_weights = sample_weights


    def __len__(
        self,
    ) -> int:

        return int(
            len(
                self.labels
            )
        )


    def __getitem__(
        self,
        index: int,
    ):

        inputs = {
            modality:
                torch.from_numpy(
                    values[
                        index
                    ]
                )
            for modality, values
            in self.arrays.items()
        }

        label = torch.tensor(
            int(
                self.labels[
                    index
                ]
            ),
            dtype=torch.long,
        )

        sample_weight = torch.tensor(
            float(
                self.sample_weights[
                    index
                ]
            ),
            dtype=torch.float32,
        )

        return (
            inputs,
            label,
            sample_weight,
        )


def make_loader(
    dataset: Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:

    generator = torch.Generator()

    generator.manual_seed(
        int(
            seed
        )
    )

    return DataLoader(
        dataset,
        batch_size=int(
            batch_size
        ),
        shuffle=bool(
            shuffle
        ),
        num_workers=0,
        drop_last=False,
        generator=generator,
        pin_memory=False,
    )


# =====================================================================
# MODEL INPUT / AMP
# =====================================================================

def model_forward(
    model: torch.nn.Module,
    inputs: Mapping[
        str,
        torch.Tensor,
    ],
) -> torch.Tensor:

    if isinstance(
        model,
        ShallowConvNet,
    ):
        return model(
            inputs[
                "EEG"
            ]
        )

    if isinstance(
        model,
        MultibranchTCN,
    ):
        return model(
            inputs
        )

    raise TypeError(
        f"Unsupported model type: {type(model)}"
    )


def amp_enabled_for_device(
    device: torch.device,
) -> bool:

    return (
        device.type
        == "cuda"
    )


def make_grad_scaler(
    device: torch.device,
):

    enabled = amp_enabled_for_device(
        device
    )

    return torch.amp.GradScaler(
        "cuda",
        enabled=enabled,
    )


# =====================================================================
# METRICS
# =====================================================================

def classification_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> dict:

    labels = np.asarray(
        labels,
        dtype=np.int64,
    )

    probabilities = np.asarray(
        probabilities,
        dtype=np.float64,
    )

    if probabilities.ndim != 2:
        raise ValueError(
            "Probability matrix must be [N,K]."
        )

    if probabilities.shape[
        0
    ] != labels.shape[
        0
    ]:
        raise ValueError(
            "Probability/label length mismatch."
        )

    n_classes = int(
        probabilities.shape[
            1
        ]
    )

    predictions = np.argmax(
        probabilities,
        axis=1,
    )

    result = {
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
                    labels=list(
                        range(
                            n_classes
                        )
                    ),
                    zero_division=0,
                )
            ),
    }

    try:

        if n_classes == 2:

            result[
                "macro_roc_auc_ovr"
            ] = float(
                roc_auc_score(
                    labels,
                    probabilities[
                        :,
                        1,
                    ],
                )
            )

            result[
                "macro_pr_auc"
            ] = float(
                average_precision_score(
                    labels,
                    probabilities[
                        :,
                        1,
                    ],
                )
            )

        else:

            result[
                "macro_roc_auc_ovr"
            ] = float(
                roc_auc_score(
                    labels,
                    probabilities,
                    labels=list(
                        range(
                            n_classes
                        )
                    ),
                    multi_class="ovr",
                    average="macro",
                )
            )

            binary = label_binarize(
                labels,
                classes=list(
                    range(
                        n_classes
                    )
                ),
            )

            result[
                "macro_pr_auc"
            ] = float(
                average_precision_score(
                    binary,
                    probabilities,
                    average="macro",
                )
            )

    except ValueError:

        result[
            "macro_roc_auc_ovr"
        ] = float(
            "nan"
        )

        result[
            "macro_pr_auc"
        ] = float(
            "nan"
        )

    return result


# =====================================================================
# VALIDATION PREDICTIONS
# =====================================================================

def predict_probabilities(
    *,
    model: torch.nn.Module,
    dataset: Dataset,
    batch_size: int,
    device: torch.device,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:

    loader = make_loader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        seed=0,
    )

    model.eval()

    labels = []
    probabilities = []

    with torch.no_grad():

        for inputs, target, _ in loader:

            inputs = {
                modality:
                    tensor.to(
                        device,
                        non_blocking=False,
                    )
                for modality, tensor
                in inputs.items()
            }

            with torch.amp.autocast(
                device_type=device.type,
                dtype=(
                    torch.float16
                    if device.type
                    == "cuda"
                    else torch.float32
                ),
                enabled=amp_enabled_for_device(
                    device
                ),
            ):

                logits = model_forward(
                    model,
                    inputs,
                )

            probs = torch.softmax(
                logits.float(),
                dim=1,
            )

            labels.append(
                target.numpy()
            )

            probabilities.append(
                probs.cpu().numpy()
            )

    return (
        np.concatenate(
            labels
        ),
        np.concatenate(
            probabilities
        ),
    )


# =====================================================================
# TRAINING CONFIGURATION
# =====================================================================

@dataclass
class TrainingSettings:
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-4
    batch_size: int = 64
    maximum_epochs: int = 100
    early_stopping_patience: int = 15
    scheduler_factor: float = 0.5
    scheduler_patience: int = 5
    minimum_learning_rate: float = 1.0e-5
    gradient_clip_norm: float = 5.0


def default_training_settings() -> TrainingSettings:

    import yaml

    payload = yaml.safe_load(
        PROTOCOL_PATH.read_text(
            encoding="utf-8"
        )
    )

    optimization = payload[
        "optimization"
    ]

    if payload[
        "protocol_revision"
    ] != "1.4":
        raise RuntimeError(
            "Expected protocol revision 1.4."
        )

    return TrainingSettings(
        learning_rate=float(
            optimization[
                "learning_rate"
            ]
        ),
        weight_decay=float(
            optimization[
                "weight_decay"
            ]
        ),
        batch_size=int(
            optimization[
                "batch_size"
            ]
        ),
        maximum_epochs=int(
            optimization[
                "maximum_epochs"
            ]
        ),
        early_stopping_patience=int(
            optimization[
                "early_stopping"
            ][
                "patience_epochs"
            ]
        ),
        scheduler_factor=float(
            optimization[
                "learning_rate_scheduler"
            ][
                "factor"
            ]
        ),
        scheduler_patience=int(
            optimization[
                "learning_rate_scheduler"
            ][
                "patience_epochs"
            ]
        ),
        minimum_learning_rate=float(
            optimization[
                "learning_rate_scheduler"
            ][
                "minimum_learning_rate"
            ]
        ),
        gradient_clip_norm=float(
            optimization[
                "gradient_clip_norm"
            ]
        ),
    )


# =====================================================================
# CHECKPOINT HELPERS
# =====================================================================

def checkpoint_paths(
    output_directory: Path,
) -> dict[
    str,
    Path,
]:

    return {
        "last":
            output_directory
            / "last.pt",

        "best":
            output_directory
            / "best.pt",

        "completed":
            output_directory
            / "completed.json",
    }


def required_checkpoint_keys() -> set[
    str
]:

    return {
        "job_identity",
        "protocol_revision",
        "registry_row_hash",
        "split_identity",
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "amp_scaler_state_dict_when_cuda",
        "epoch",
        "best_validation_balanced_accuracy",
        "best_validation_macro_f1",
        "best_epoch",
        "early_stopping_bad_epoch_count",
        "learning_rate",
        "training_normalizer",
        "python_random_state",
        "numpy_random_state",
        "torch_cpu_rng_state",
        "torch_cuda_rng_state_all_devices_when_cuda",
        "training_seed",
    }


def validate_checkpoint_identity(
    *,
    checkpoint: Mapping,
    expected: Mapping,
) -> None:

    missing = (
        required_checkpoint_keys()
        - set(
            checkpoint.keys()
        )
    )

    if missing:
        raise RuntimeError(
            f"Checkpoint is missing required keys: {sorted(missing)}"
        )

    for key, value in expected.items():

        if checkpoint.get(
            key
        ) != value:

            raise RuntimeError(
                f"Checkpoint identity mismatch for {key}: "
                f"{checkpoint.get(key)!r} != {value!r}"
            )


# =====================================================================
# PROGRESS DISPLAY
# =====================================================================

class EpochProgress:

    def __init__(
        self,
        *,
        total: int,
        description: str,
        enabled: bool,
    ) -> None:

        self.enabled = bool(
            enabled
        )

        self.total = int(
            total
        )

        self.start_time = time.time()

        self.count = 0

        self.bar = None

        if self.enabled:

            try:
                from tqdm import tqdm

                self.bar = tqdm(
                    total=self.total,
                    desc=description,
                    unit="epoch",
                    dynamic_ncols=True,
                    leave=False,
                )

            except Exception:
                self.bar = None


    def update(
        self,
        *,
        context: str,
    ) -> None:

        self.count += 1

        if not self.enabled:
            return

        if self.bar is not None:

            self.bar.set_postfix_str(
                context
            )

            self.bar.update(
                1
            )

            return

        elapsed = max(
            time.time()
            - self.start_time,
            1.0e-9,
        )

        rate = (
            self.count
            / elapsed
        )

        eta = (
            (
                self.total
                - self.count
            )
            / rate
            if rate > 0
            else float(
                "inf"
            )
        )

        print(
            f"epoch {self.count}/{self.total} | "
            f"elapsed={elapsed:.1f}s | "
            f"ETA={eta:.1f}s | "
            f"{context}",
            flush=True,
        )


    def close(
        self,
    ) -> None:

        if self.bar is not None:
            self.bar.close()


# =====================================================================
# SUPERVISED TRAINING
# =====================================================================

def train_supervised(
    *,
    model: torch.nn.Module,
    train_arrays: Mapping[
        str,
        np.ndarray,
    ],
    train_labels: np.ndarray,
    validation_arrays: Mapping[
        str,
        np.ndarray,
    ],
    validation_labels: np.ndarray,
    n_classes: int,
    training_seed: int,
    output_directory: Path,
    job_identity: str,
    registry_row_hash: str,
    split_identity: str,
    settings: TrainingSettings | None = None,
    device: torch.device | None = None,
    sample_weights: np.ndarray | None = None,
    resume: bool = True,
    final_head_only: bool = False,
    show_progress: bool = True,
    interrupt_after_epoch: int | None = None,
) -> dict:

    if settings is None:
        settings = default_training_settings()

    if device is None:
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    output_directory = Path(
        output_directory
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    paths = checkpoint_paths(
        output_directory
    )

    if (
        paths[
            "completed"
        ].is_file()
        and resume
    ):

        completed = json.loads(
            paths[
                "completed"
            ].read_text(
                encoding="utf-8"
            )
        )

        expected_completed_identity = {
            "job_identity":
                job_identity,

            "protocol_revision":
                "1.4",

            "registry_row_hash":
                registry_row_hash,

            "split_identity":
                split_identity,

            "training_seed":
                int(
                    training_seed
                ),
        }

        for key, value in expected_completed_identity.items():

            if completed.get(
                key
            ) != value:
                raise RuntimeError(
                    f"Completion-manifest identity mismatch: {key}"
                )

        best_path = Path(
            completed[
                "best_checkpoint"
            ]
        )

        if not best_path.is_absolute():
            best_path = (
                ROOT
                / best_path
            )

        if not best_path.is_file():
            raise RuntimeError(
                "Completion manifest points to missing best checkpoint."
            )

        if file_sha256(
            best_path
        ) != completed[
            "best_checkpoint_sha256"
        ]:
            raise RuntimeError(
                "Completed-job best-checkpoint hash mismatch."
            )

        return {
            "status":
                "already_completed",

            "best_epoch":
                int(
                    completed[
                        "best_epoch"
                    ]
                ),

            "best_validation_balanced_accuracy":
                float(
                    completed[
                        "best_validation_balanced_accuracy"
                    ]
                ),

            "best_validation_macro_f1":
                float(
                    completed[
                        "best_validation_macro_f1"
                    ]
                ),

            "output_directory":
                str(
                    output_directory
                ),
        }


    set_global_determinism(
        training_seed
    )

    train_labels = np.asarray(
        train_labels,
        dtype=np.int64,
    )

    validation_labels = np.asarray(
        validation_labels,
        dtype=np.int64,
    )

    if len(
        train_labels
    ) == 0:
        raise ValueError(
            "Training partition is empty."
        )

    if len(
        validation_labels
    ) == 0:
        raise ValueError(
            "Validation partition is empty."
        )


    if final_head_only:

        trainable_names = configure_final_head_only(
            model
        )

    else:

        trainable_names = [
            name
            for name, parameter
            in model.named_parameters()
            if parameter.requires_grad
        ]


    model = model.to(
        device
    )

    start_epoch = 1

    best_balanced_accuracy = -float(
        "inf"
    )

    best_macro_f1 = -float(
        "inf"
    )

    best_epoch = 0

    bad_epochs = 0


    checkpoint = None

    if (
        resume
        and paths[
            "last"
        ].is_file()
    ):

        checkpoint = torch.load(
            paths[
                "last"
            ],
            map_location="cpu",
            weights_only=False,
        )


        validate_checkpoint_identity(
            checkpoint=checkpoint,
            expected={
                "job_identity":
                    job_identity,

                "protocol_revision":
                    "1.4",

                "registry_row_hash":
                    registry_row_hash,

                "split_identity":
                    split_identity,

                "training_seed":
                    int(
                        training_seed
                    ),
            },
        )


        normalizer = ChannelZScoreNormalizer.from_state_dict(
            checkpoint[
                "training_normalizer"
            ]
        )

    else:

        normalizer = ChannelZScoreNormalizer().fit(
            train_arrays
        )


    normalized_train = normalizer.transform(
        train_arrays
    )

    normalized_validation = normalizer.transform(
        validation_arrays
    )


    train_dataset = ArrayClassificationDataset(
        arrays=normalized_train,
        labels=train_labels,
        sample_weights=sample_weights,
    )

    validation_dataset = ArrayClassificationDataset(
        arrays=normalized_validation,
        labels=validation_labels,
    )


    class_weights = torch.as_tensor(
        inverse_frequency_class_weights(
            train_labels,
            n_classes=n_classes,
        ),
        dtype=torch.float32,
        device=device,
    )


    optimizer = torch.optim.AdamW(
        [
            parameter
            for parameter
            in model.parameters()
            if parameter.requires_grad
        ],
        lr=float(
            settings.learning_rate
        ),
        weight_decay=float(
            settings.weight_decay
        ),
    )


    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=float(
            settings.scheduler_factor
        ),
        patience=int(
            settings.scheduler_patience
        ),
        min_lr=float(
            settings.minimum_learning_rate
        ),
    )


    scaler = make_grad_scaler(
        device
    )


    if checkpoint is not None:

        model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

        optimizer.load_state_dict(
            checkpoint[
                "optimizer_state_dict"
            ]
        )

        scheduler.load_state_dict(
            checkpoint[
                "scheduler_state_dict"
            ]
        )

        scaler_state = checkpoint.get(
            "amp_scaler_state_dict_when_cuda"
        )

        if (
            scaler_state is not None
            and amp_enabled_for_device(
                device
            )
        ):
            scaler.load_state_dict(
                scaler_state
            )

        best_balanced_accuracy = float(
            checkpoint[
                "best_validation_balanced_accuracy"
            ]
        )

        best_macro_f1 = float(
            checkpoint[
                "best_validation_macro_f1"
            ]
        )

        best_epoch = int(
            checkpoint[
                "best_epoch"
            ]
        )

        bad_epochs = int(
            checkpoint[
                "early_stopping_bad_epoch_count"
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

        restore_rng_state(
            checkpoint
        )


    total_remaining_epochs = max(
        0,
        int(
            settings.maximum_epochs
        )
        - start_epoch
        + 1,
    )

    progress = EpochProgress(
        total=total_remaining_epochs,
        description=job_identity,
        enabled=show_progress,
    )


    interrupted = False

    final_epoch = (
        start_epoch
        - 1
    )


    try:

        for epoch in range(
            start_epoch,
            int(
                settings.maximum_epochs
            )
            + 1,
        ):

            final_epoch = epoch

            model.train()

            train_loader = make_loader(
                train_dataset,
                batch_size=int(
                    settings.batch_size
                ),
                shuffle=True,
                seed=dataloader_seed(
                    training_seed,
                    epoch,
                ),
            )


            running_weighted_loss = 0.0
            running_weight = 0.0


            for inputs, targets, batch_sample_weights in train_loader:

                inputs = {
                    modality:
                        tensor.to(
                            device
                        )
                    for modality, tensor
                    in inputs.items()
                }

                targets = targets.to(
                    device
                )

                batch_sample_weights = batch_sample_weights.to(
                    device
                )


                optimizer.zero_grad(
                    set_to_none=True
                )


                with torch.amp.autocast(
                    device_type=device.type,
                    dtype=(
                        torch.float16
                        if device.type
                        == "cuda"
                        else torch.float32
                    ),
                    enabled=amp_enabled_for_device(
                        device
                    ),
                ):

                    logits = model_forward(
                        model,
                        inputs,
                    )

                    per_sample_loss = F.cross_entropy(
                        logits,
                        targets,
                        weight=class_weights,
                        reduction="none",
                    )

                    weighted = (
                        per_sample_loss
                        * batch_sample_weights
                    )

                    loss = (
                        weighted.sum()
                        / batch_sample_weights.sum()
                    )


                scaler.scale(
                    loss
                ).backward()


                scaler.unscale_(
                    optimizer
                )

                torch.nn.utils.clip_grad_norm_(
                    [
                        parameter
                        for parameter
                        in model.parameters()
                        if parameter.requires_grad
                    ],
                    max_norm=float(
                        settings.gradient_clip_norm
                    ),
                )


                scaler.step(
                    optimizer
                )

                scaler.update()


                running_weighted_loss += float(
                    weighted.detach().sum().cpu()
                )

                running_weight += float(
                    batch_sample_weights.detach().sum().cpu()
                )


            validation_y, validation_probabilities = (
                predict_probabilities(
                    model=model,
                    dataset=validation_dataset,
                    batch_size=int(
                        settings.batch_size
                    ),
                    device=device,
                )
            )


            metrics = classification_metrics(
                validation_y,
                validation_probabilities,
            )


            validation_balanced_accuracy = float(
                metrics[
                    "balanced_accuracy"
                ]
            )

            validation_macro_f1 = float(
                metrics[
                    "macro_f1"
                ]
            )


            scheduler.step(
                validation_balanced_accuracy
            )


            improved = (
                validation_balanced_accuracy
                > best_balanced_accuracy
                + 1.0e-12
            ) or (
                math.isclose(
                    validation_balanced_accuracy,
                    best_balanced_accuracy,
                    abs_tol=1.0e-12,
                )
                and (
                    validation_macro_f1
                    > best_macro_f1
                    + 1.0e-12
                )
            )


            if improved:

                best_balanced_accuracy = (
                    validation_balanced_accuracy
                )

                best_macro_f1 = (
                    validation_macro_f1
                )

                best_epoch = epoch
                bad_epochs = 0

            else:

                bad_epochs += 1


            rng_state = capture_rng_state()


            checkpoint_payload = {
                "job_identity":
                    job_identity,

                "protocol_revision":
                    "1.4",

                "registry_row_hash":
                    registry_row_hash,

                "split_identity":
                    split_identity,

                "model_state_dict":
                    {
                        key:
                            value.detach().cpu()
                        for key, value
                        in model.state_dict().items()
                    },

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "scheduler_state_dict":
                    scheduler.state_dict(),

                "amp_scaler_state_dict_when_cuda":
                    (
                        scaler.state_dict()
                        if amp_enabled_for_device(
                            device
                        )
                        else None
                    ),

                "epoch":
                    epoch,

                "best_validation_balanced_accuracy":
                    best_balanced_accuracy,

                "best_validation_macro_f1":
                    best_macro_f1,

                "best_epoch":
                    best_epoch,

                "early_stopping_bad_epoch_count":
                    bad_epochs,

                "learning_rate":
                    float(
                        optimizer.param_groups[
                            0
                        ][
                            "lr"
                        ]
                    ),

                "training_normalizer":
                    normalizer.state_dict(),

                "python_random_state":
                    rng_state[
                        "python_random_state"
                    ],

                "numpy_random_state":
                    rng_state[
                        "numpy_random_state"
                    ],

                "torch_cpu_rng_state":
                    rng_state[
                        "torch_cpu_rng_state"
                    ],

                "torch_cuda_rng_state_all_devices_when_cuda":
                    rng_state[
                        "torch_cuda_rng_state_all_devices_when_cuda"
                    ],

                "training_seed":
                    int(
                        training_seed
                    ),

                "trainable_parameter_names":
                    trainable_names,

                "validation_metrics":
                    metrics,

                "mean_training_loss":
                    (
                        running_weighted_loss
                        / max(
                            running_weight,
                            1.0e-12,
                        )
                    ),
            }


            atomic_torch_save(
                paths[
                    "last"
                ],
                checkpoint_payload,
            )


            if improved:

                atomic_torch_save(
                    paths[
                        "best"
                    ],
                    checkpoint_payload,
                )


            learning_rate = float(
                optimizer.param_groups[
                    0
                ][
                    "lr"
                ]
            )


            progress.update(
                context=(
                    f"valBA={validation_balanced_accuracy:.4f} "
                    f"valF1={validation_macro_f1:.4f} "
                    f"best={best_balanced_accuracy:.4f} "
                    f"lr={learning_rate:.2e} "
                    f"patience={bad_epochs}/"
                    f"{settings.early_stopping_patience}"
                )
            )


            if (
                interrupt_after_epoch is not None
                and epoch
                >= int(
                    interrupt_after_epoch
                )
            ):

                interrupted = True
                break


            if (
                bad_epochs
                >= int(
                    settings.early_stopping_patience
                )
            ):
                break

    finally:

        progress.close()


    if interrupted:

        return {
            "status":
                "interrupted",

            "last_epoch":
                final_epoch,

            "best_epoch":
                best_epoch,

            "best_validation_balanced_accuracy":
                best_balanced_accuracy,

            "best_validation_macro_f1":
                best_macro_f1,

            "output_directory":
                str(
                    output_directory
                ),
        }


    if not paths[
        "best"
    ].is_file():

        raise RuntimeError(
            "Training finished without a best checkpoint."
        )


    completed = {
        "job_identity":
            job_identity,

        "protocol_revision":
            "1.4",

        "registry_row_hash":
            registry_row_hash,

        "split_identity":
            split_identity,

        "training_seed":
            int(
                training_seed
            ),

        "status":
            "completed",

        "last_epoch":
            final_epoch,

        "best_epoch":
            best_epoch,

        "best_validation_balanced_accuracy":
            best_balanced_accuracy,

        "best_validation_macro_f1":
            best_macro_f1,

        "best_checkpoint":
            str(
                paths[
                    "best"
                ]
            ),

        "best_checkpoint_sha256":
            file_sha256(
                paths[
                    "best"
                ]
            ),

        "last_checkpoint":
            str(
                paths[
                    "last"
                ]
            ),

        "last_checkpoint_sha256":
            file_sha256(
                paths[
                    "last"
                ]
            ),
    }


    atomic_json_write(
        paths[
            "completed"
        ],
        completed,
    )


    return {
        "status":
            "completed",

        "last_epoch":
            final_epoch,

        "best_epoch":
            best_epoch,

        "best_validation_balanced_accuracy":
            best_balanced_accuracy,

        "best_validation_macro_f1":
            best_macro_f1,

        "output_directory":
            str(
                output_directory
            ),
    }


# =====================================================================
# REGISTRY EXPANSION
# =====================================================================

def normalize_registry_scalar(
    value,
):

    if pd.isna(
        value
    ):
        return None

    if isinstance(
        value,
        np.generic,
    ):
        return value.item()

    return value


def registry_row_payload(
    row: Mapping,
) -> dict:

    return {
        str(key):
            normalize_registry_scalar(
                value
            )
        for key, value
        in dict(
            row
        ).items()
    }


def expand_execution_registry(
    registry: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Expand a frozen execution registry into fit/adaptation operations.

    Supported registry generations
    ------------------------------
    v2 historical registry:
        3,000 rows -> 3,936 operations.

        All strict nested-LOSO model/path rows contain candidate
        durations [8, 16, 24, 32].

    v3 structural-feasibility registry:
        2,532 rows -> 3,000 operations.

        Pupil-containing MultibranchTCN nested rows contain only [8].
        Pupil-free nested rows retain [8, 16, 24, 32].
        Pupil-containing subject-adaptive rows are absent.

    The function does not infer which generation it received from a
    global protocol version. It validates the supplied registry itself.
    """

    if registry is None:

        # Historical default is intentionally retained so older audits
        # and regression tests remain reproducible. The final real-data
        # launcher must explicitly supply registry v3.
        registry = pd.read_csv(
            REGISTRY_V2,
            low_memory=False,
        )


    rows = []


    for _, source_row in registry.iterrows():

        base = registry_row_payload(
            source_row.to_dict()
        )


        stage = str(
            base[
                "stage"
            ]
        )

        protocol = str(
            base[
                "protocol"
            ]
        )

        dataset = str(
            base[
                "dataset"
            ]
        )

        split_id = str(
            base[
                "split_id"
            ]
        )

        model = str(
            base[
                "model"
            ]
        )

        path = str(
            base[
                "path"
            ]
        )

        seed = int(
            base[
                "seed"
            ]
        )


        if path not in LOCKED_MODALITY_PATHS:

            raise RuntimeError(
                f"Unknown locked modality path: {path}"
            )


        identity_prefix = (
            f"{stage}/"
            f"{protocol}/"
            f"{dataset}/"
            f"{split_id}/"
            f"{model}/"
            f"{path}/"
            f"seed_{seed}"
        )


        row_hash = canonical_json_hash(
            base
        )


        # ========================================================
        # STRICT NESTED LOSO
        # ========================================================

        if stage == "internal_nested_loso":

            raw_durations = base.get(
                "candidate_durations_seconds"
            )


            if raw_durations is None:

                raise RuntimeError(
                    "Nested LOSO row is missing candidate durations."
                )


            durations = [
                int(
                    value
                )
                for value
                in str(
                    raw_durations
                ).split(";")
                if str(
                    value
                ).strip()
            ]


            is_pupil_tcn = (
                model
                == "MultibranchTCN"
                and "Pupil"
                in LOCKED_MODALITY_PATHS[
                    path
                ]
            )


            allowed_duration_sets = [
                [
                    8,
                    16,
                    24,
                    32,
                ]
            ]


            if is_pupil_tcn:

                # Historical registry v2 contains all four candidates.
                # Final registry v3 contains structurally feasible 8 s.
                allowed_duration_sets.append(
                    [
                        8
                    ]
                )


            if durations not in allowed_duration_sets:

                raise RuntimeError(
                    f"Unexpected nested durations for "
                    f"{model}/{path}: {durations}; "
                    f"allowed={allowed_duration_sets}"
                )


            for duration in durations:

                operation = copy.deepcopy(
                    base
                )

                operation[
                    "duration_candidate_seconds"
                ] = int(
                    duration
                )

                operation[
                    "fit_operation_type"
                ] = "global_training"

                operation[
                    "operation_id"
                ] = (
                    f"{identity_prefix}/"
                    f"duration_{duration}s"
                )

                operation[
                    "registry_row_hash"
                ] = row_hash

                rows.append(
                    operation
                )


        # ========================================================
        # SUBJECT-ADAPTIVE
        # ========================================================

        elif stage == "internal_subject_adaptive":

            raw_budget = base.get(
                "calibration_budget_seconds_per_class"
            )


            if raw_budget is None:

                raise RuntimeError(
                    "Adaptive row is missing calibration budget."
                )


            budget = int(
                float(
                    raw_budget
                )
            )


            if budget not in {
                30,
                60,
                120,
            }:

                raise RuntimeError(
                    f"Unexpected adaptive budget: {budget}"
                )


            dependency = base.get(
                "dependency"
            )


            if (
                dependency is None
                or not str(
                    dependency
                ).startswith(
                    "internal_nested_loso/"
                )
            ):

                raise RuntimeError(
                    "Adaptive row has invalid nested-LOSO dependency."
                )


            operation = copy.deepcopy(
                base
            )

            operation[
                "duration_candidate_seconds"
            ] = None

            operation[
                "calibration_budget_seconds_per_class"
            ] = budget

            operation[
                "fit_operation_type"
            ] = "final_head_adaptation"

            operation[
                "operation_id"
            ] = (
                f"{identity_prefix}/"
                f"budget_{budget}s_per_class"
            )

            operation[
                "registry_row_hash"
            ] = row_hash

            rows.append(
                operation
            )


        # ========================================================
        # CONVENTIONAL + BBBD
        # ========================================================

        else:

            if stage not in {
                "internal_conventional",
                "bbbd_within_experiment",
                "bbbd_cross_experiment",
            }:

                raise RuntimeError(
                    f"Unknown execution stage: {stage}"
                )


            operation = copy.deepcopy(
                base
            )

            operation[
                "duration_candidate_seconds"
            ] = 4

            operation[
                "fit_operation_type"
            ] = "global_training"

            operation[
                "operation_id"
            ] = identity_prefix

            operation[
                "registry_row_hash"
            ] = row_hash

            rows.append(
                operation
            )


    operations = pd.DataFrame(
        rows
    )


    # ============================================================
    # IDENTIFY REGISTRY GENERATION FROM ITS OWN ROW CONTRACT
    # ============================================================

    registry_stage_counts = (
        registry[
            "stage"
        ]
        .astype(str)
        .value_counts()
        .sort_index()
        .to_dict()
    )


    historical_v2_stage_counts = {
        "bbbd_cross_experiment":
            48,

        "bbbd_within_experiment":
            864,

        "internal_conventional":
            840,

        "internal_nested_loso":
            312,

        "internal_subject_adaptive":
            936,
    }


    final_v3_stage_counts = {
        "bbbd_cross_experiment":
            48,

        "bbbd_within_experiment":
            864,

        "internal_conventional":
            840,

        "internal_nested_loso":
            312,

        "internal_subject_adaptive":
            468,
    }


    if (
        len(
            registry
        )
        == 3000
        and registry_stage_counts
        == historical_v2_stage_counts
    ):

        registry_generation = "v2"

        expected_operation_counts = {
            "bbbd_cross_experiment":
                48,

            "bbbd_within_experiment":
                864,

            "internal_conventional":
                840,

            "internal_nested_loso":
                1248,

            "internal_subject_adaptive":
                936,
        }

        expected_total = 3936


    elif (
        len(
            registry
        )
        == 2532
        and registry_stage_counts
        == final_v3_stage_counts
    ):

        registry_generation = "v3"

        expected_operation_counts = {
            "bbbd_cross_experiment":
                48,

            "bbbd_within_experiment":
                864,

            "internal_conventional":
                840,

            "internal_nested_loso":
                780,

            "internal_subject_adaptive":
                468,
        }

        expected_total = 3000


    else:

        raise RuntimeError(
            "Unknown execution-registry generation: "
            f"rows={len(registry)}, "
            f"stage_counts={registry_stage_counts}"
        )


    actual_operation_counts = (
        operations[
            "stage"
        ]
        .astype(str)
        .value_counts()
        .sort_index()
        .to_dict()
    )


    if (
        actual_operation_counts
        != expected_operation_counts
    ):

        raise RuntimeError(
            f"{registry_generation} expanded-operation counts "
            "do not match the frozen contract:\n"
            f"expected={expected_operation_counts}\n"
            f"actual={actual_operation_counts}"
        )


    if len(
        operations
    ) != expected_total:

        raise RuntimeError(
            f"{registry_generation}: expected "
            f"{expected_total:,} operations, "
            f"found {len(operations):,}."
        )


    duplicated = operations[
        "operation_id"
    ].duplicated(
        keep=False
    )


    if duplicated.any():

        values = (
            operations.loc[
                duplicated,
                "operation_id",
            ]
            .astype(str)
            .tolist()
        )

        raise RuntimeError(
            "Duplicate operation IDs remain: "
            f"{values[:25]}"
        )


    for row in operations.itertuples(
        index=False
    ):

        protocol_token = (
            f"/{row.protocol}/"
        )

        if protocol_token not in str(
            row.operation_id
        ):

            raise RuntimeError(
                "Operation ID lost protocol identity: "
                f"{row.operation_id}"
            )


    return operations


# =====================================================================
# REAL-EXECUTION GATE
# =====================================================================

def assert_real_training_gate_open() -> None:

    import yaml

    protocol = yaml.safe_load(
        PROTOCOL_PATH.read_text(
            encoding="utf-8"
        )
    )

    if protocol[
        "protocol_revision"
    ] != "1.4":
        raise RuntimeError(
            "Expected protocol revision 1.4."
        )

    if not bool(
        protocol[
            "execution_gate"
        ][
            "real_training_allowed_now"
        ]
    ):

        raise RuntimeError(
            "REAL TRAINING IS BLOCKED. "
            "Complete the registry/data-binding preflight and "
            "explicitly open the frozen execution gate first."
        )
