
"""Frozen deep-learning baseline architectures.

Scientific scope
----------------
ShallowConvNet
    Canonical EEG-only auxiliary baseline.

MultibranchTCN
    Lightweight temporal representation-learning baseline supporting the
    seven locked EEG/ECG/Pupil modality paths.

This module contains model definitions only. It performs no data loading,
split construction, fitting, evaluation, or performance-based selection.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F


LOCKED_EEG_CHANNELS = (
    "AF7",
    "Fp1",
    "Fpz",
    "Fp2",
    "AF8",
    "O1",
    "POz",
    "O2",
)

MODALITY_INPUT_CHANNELS = {
    "EEG": 8,
    "ECG": 1,
    "Pupil": 1,
}

LOCKED_MODALITY_PATHS = {
    "EEG": ("EEG",),
    "ECG": ("ECG",),
    "Pupil": ("Pupil",),
    "ECG_EEG": ("ECG", "EEG"),
    "ECG_Pupil": ("ECG", "Pupil"),
    "EEG_Pupil": ("EEG", "Pupil"),
    "ECG_EEG_Pupil": ("ECG", "EEG", "Pupil"),
}


class SafeLog(nn.Module):
    """Numerically safe natural logarithm."""

    def __init__(self, minimum: float = 1.0e-6) -> None:
        super().__init__()

        if minimum <= 0:
            raise ValueError(
                "SafeLog minimum must be positive."
            )

        self.minimum = float(minimum)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return torch.log(
            torch.clamp(
                x,
                min=self.minimum,
            )
        )


class Square(nn.Module):
    """Element-wise square operation."""

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return x * x


class ShallowConvNet(nn.Module):
    """Frozen EEG-only ShallowConvNet baseline.

    Expected input
    --------------
    Either:
        [batch, eeg_channels, time]
    or:
        [batch, 1, eeg_channels, time]

    The constructor requires the temporal length so that the final
    flattened classifier is fixed for the requested window duration.
    """

    def __init__(
        self,
        *,
        n_classes: int,
        n_times: int,
        eeg_channels: int = 8,
        temporal_filters: int = 40,
        temporal_kernel_samples: int = 13,
        spatial_filters: int = 40,
        pool_samples: int = 38,
        pool_stride_samples: int = 8,
        dropout: float = 0.5,
        log_clamp_min: float = 1.0e-6,
    ) -> None:
        super().__init__()

        if n_classes < 2:
            raise ValueError(
                "n_classes must be at least 2."
            )

        if eeg_channels != 8:
            raise ValueError(
                "Frozen ShallowConvNet requires exactly 8 EEG channels."
            )

        if temporal_filters != 40:
            raise ValueError(
                "Frozen ShallowConvNet requires 40 temporal filters."
            )

        if spatial_filters != 40:
            raise ValueError(
                "Frozen ShallowConvNet requires 40 spatial filters."
            )

        if spatial_filters != temporal_filters:
            raise ValueError(
                "Frozen depthwise spatial filtering requires equal "
                "temporal and spatial filter counts."
            )

        if n_times <= (
            temporal_kernel_samples
            + pool_samples
        ):
            raise ValueError(
                "Temporal input is too short for the frozen architecture."
            )

        self.n_classes = int(n_classes)
        self.n_times = int(n_times)
        self.eeg_channels = int(eeg_channels)

        self.temporal_conv = nn.Conv2d(
            in_channels=1,
            out_channels=temporal_filters,
            kernel_size=(
                1,
                temporal_kernel_samples,
            ),
            bias=False,
        )

        # One spatial filter per temporal filter, spanning all EEG channels.
        self.spatial_conv = nn.Conv2d(
            in_channels=temporal_filters,
            out_channels=spatial_filters,
            kernel_size=(
                eeg_channels,
                1,
            ),
            groups=temporal_filters,
            bias=False,
        )

        self.batch_norm = nn.BatchNorm2d(
            spatial_filters,
            momentum=0.1,
            affine=True,
        )

        self.square = Square()

        self.average_pool = nn.AvgPool2d(
            kernel_size=(
                1,
                pool_samples,
            ),
            stride=(
                1,
                pool_stride_samples,
            ),
        )

        self.safe_log = SafeLog(
            minimum=log_clamp_min
        )

        self.dropout = nn.Dropout(
            p=dropout
        )

        feature_dimension = self._infer_feature_dimension()

        self.classifier = nn.Linear(
            feature_dimension,
            n_classes,
        )

    def _standardize_input(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        if x.ndim == 3:
            x = x.unsqueeze(1)

        if x.ndim != 4:
            raise ValueError(
                "ShallowConvNet input must be [B,C,T] or [B,1,C,T]."
            )

        if x.shape[1] != 1:
            raise ValueError(
                "ShallowConvNet expects singleton convolution input channel."
            )

        if x.shape[2] != self.eeg_channels:
            raise ValueError(
                f"Expected {self.eeg_channels} EEG channels, "
                f"received {x.shape[2]}."
            )

        if x.shape[3] != self.n_times:
            raise ValueError(
                f"Expected {self.n_times} temporal samples, "
                f"received {x.shape[3]}."
            )

        return x

    def _forward_features(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        x = self.temporal_conv(x)
        x = self.spatial_conv(x)
        x = self.batch_norm(x)
        x = self.square(x)
        x = self.average_pool(x)
        x = self.safe_log(x)
        x = self.dropout(x)

        return torch.flatten(
            x,
            start_dim=1,
        )

    def _infer_feature_dimension(
        self,
    ) -> int:

        previous_training_state = self.training
        self.eval()

        with torch.no_grad():
            dummy = torch.zeros(
                2,
                1,
                self.eeg_channels,
                self.n_times,
                dtype=torch.float32,
            )

            features = self._forward_features(
                dummy
            )

        self.train(
            previous_training_state
        )

        return int(
            features.shape[1]
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        x = self._standardize_input(
            x
        )

        features = self._forward_features(
            x
        )

        return self.classifier(
            features
        )


class CausalConv1d(nn.Module):
    """Conv1d using left-only padding."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int,
        dilation: int,
        bias: bool = True,
    ) -> None:
        super().__init__()

        if kernel_size < 1:
            raise ValueError(
                "kernel_size must be positive."
            )

        if dilation < 1:
            raise ValueError(
                "dilation must be positive."
            )

        self.left_padding = (
            (kernel_size - 1)
            * dilation
        )

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=0,
            bias=bias,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        x = F.pad(
            x,
            (
                self.left_padding,
                0,
            ),
        )

        return self.conv(
            x
        )


class ResidualTCNBlock(nn.Module):
    """Frozen two-convolution causal residual TCN block."""

    def __init__(
        self,
        channels: int,
        *,
        kernel_size: int,
        dilation: int,
        group_count: int,
        dropout: float,
    ) -> None:
        super().__init__()

        if channels % group_count != 0:
            raise ValueError(
                "channels must be divisible by group_count."
            )

        self.conv1 = CausalConv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            dilation=dilation,
            bias=True,
        )

        self.norm1 = nn.GroupNorm(
            num_groups=group_count,
            num_channels=channels,
        )

        self.conv2 = CausalConv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            dilation=dilation,
            bias=True,
        )

        self.norm2 = nn.GroupNorm(
            num_groups=group_count,
            num_channels=channels,
        )

        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(
            p=dropout
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        residual = x

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = x + residual
        x = self.activation(x)

        return x


class TemporalBranch(nn.Module):
    """One frozen modality-specific temporal branch."""

    def __init__(
        self,
        *,
        input_channels: int,
        branch_width: int = 32,
        dilations: Sequence[int] = (
            1,
            2,
            4,
            8,
        ),
        kernel_size: int = 3,
        group_count: int = 4,
        dropout: float = 0.20,
    ) -> None:
        super().__init__()

        self.input_channels = int(
            input_channels
        )

        self.branch_width = int(
            branch_width
        )

        self.input_projection = nn.Conv1d(
            self.input_channels,
            self.branch_width,
            kernel_size=1,
            bias=True,
        )

        self.blocks = nn.ModuleList(
            [
                ResidualTCNBlock(
                    self.branch_width,
                    kernel_size=kernel_size,
                    dilation=int(dilation),
                    group_count=group_count,
                    dropout=dropout,
                )
                for dilation in dilations
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        if x.ndim != 3:
            raise ValueError(
                "TCN branch input must be [batch,channels,time]."
            )

        if x.shape[1] != self.input_channels:
            raise ValueError(
                f"Expected {self.input_channels} channels, "
                f"received {x.shape[1]}."
            )

        x = self.input_projection(
            x
        )

        for block in self.blocks:
            x = block(x)

        # Frozen branch readout: global temporal average pooling.
        return x.mean(
            dim=-1
        )


class MultibranchTCN(nn.Module):
    """Frozen seven-path multimodal TCN architecture."""

    def __init__(
        self,
        *,
        path: str,
        n_classes: int,
        branch_width: int = 32,
        fusion_hidden_dimension: int = 64,
        branch_dropout: float = 0.20,
        fusion_dropout: float = 0.30,
        kernel_size: int = 3,
        dilations: Sequence[int] = (
            1,
            2,
            4,
            8,
        ),
        group_count: int = 4,
    ) -> None:
        super().__init__()

        if path not in LOCKED_MODALITY_PATHS:
            raise ValueError(
                f"Unknown modality path: {path}"
            )

        if n_classes < 2:
            raise ValueError(
                "n_classes must be at least 2."
            )

        self.path = str(path)
        self.n_classes = int(n_classes)

        self.active_modalities = (
            LOCKED_MODALITY_PATHS[
                self.path
            ]
        )

        self.branches = nn.ModuleDict(
            {
                modality:
                    TemporalBranch(
                        input_channels=
                            MODALITY_INPUT_CHANNELS[
                                modality
                            ],
                        branch_width=branch_width,
                        dilations=dilations,
                        kernel_size=kernel_size,
                        group_count=group_count,
                        dropout=branch_dropout,
                    )
                for modality
                in self.active_modalities
            }
        )

        fused_dimension = (
            branch_width
            * len(
                self.active_modalities
            )
        )

        self.fusion = nn.Sequential(
            nn.Linear(
                fused_dimension,
                fusion_hidden_dimension,
            ),
            nn.ReLU(),
            nn.Dropout(
                p=fusion_dropout
            ),
            nn.Linear(
                fusion_hidden_dimension,
                n_classes,
            ),
        )

    def forward(
        self,
        inputs: Mapping[
            str,
            torch.Tensor,
        ],
    ) -> torch.Tensor:

        missing = [
            modality
            for modality
            in self.active_modalities
            if modality not in inputs
        ]

        if missing:
            raise ValueError(
                f"Missing required modalities: {missing}"
            )

        embeddings = []

        batch_size = None

        for modality in self.active_modalities:

            tensor = inputs[
                modality
            ]

            if batch_size is None:
                batch_size = int(
                    tensor.shape[0]
                )

            elif int(
                tensor.shape[0]
            ) != batch_size:
                raise ValueError(
                    "All modality branches must share the same batch size."
                )

            embeddings.append(
                self.branches[
                    modality
                ](
                    tensor
                )
            )

        fused = torch.cat(
            embeddings,
            dim=1,
        )

        return self.fusion(
            fused
        )


def count_trainable_parameters(
    model: nn.Module,
) -> int:
    """Return the number of trainable parameters."""

    return sum(
        parameter.numel()
        for parameter
        in model.parameters()
        if parameter.requires_grad
    )
