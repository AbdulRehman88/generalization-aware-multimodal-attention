
"""Synthetic-only architecture smoke test.

No real participant, recording, feature, label, or BBBD data are loaded.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.models.deep_learning_baselines import (
    LOCKED_MODALITY_PATHS,
    MultibranchTCN,
    ShallowConvNet,
    count_trainable_parameters,
)


ROOT = Path(__file__).resolve().parents[2]

PROTOCOL = (
    ROOT
    / "configs/deep_learning_baselines.yaml"
)

OUT_JSON = (
    ROOT
    / "_research_audit/deep_learning_architecture_smoke_v2.json"
)

OUT_MD = (
    ROOT
    / "_research_audit/deep_learning_architecture_smoke_v2.md"
)


def sha256(
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


def set_seed(
    seed: int,
) -> None:

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            seed
        )


def one_step(
    model: nn.Module,
    inputs,
    *,
    classes: int,
    device: torch.device,
):

    model = model.to(
        device
    )

    model.train()

    if isinstance(
        inputs,
        dict,
    ):
        moved = {
            key:
                value.to(
                    device
                )
            for key, value
            in inputs.items()
        }

        batch_size = next(
            iter(
                moved.values()
            )
        ).shape[0]

        logits = model(
            moved
        )

    else:
        moved = inputs.to(
            device
        )

        batch_size = moved.shape[0]

        logits = model(
            moved
        )

    target = torch.arange(
        batch_size,
        device=device,
        dtype=torch.long,
    ) % classes

    loss = nn.CrossEntropyLoss()(
        logits,
        target,
    )

    loss.backward()

    gradients = [
        parameter.grad
        for parameter
        in model.parameters()
        if (
            parameter.requires_grad
            and parameter.grad
            is not None
        )
    ]

    if not gradients:
        raise RuntimeError(
            "No gradients were produced."
        )

    if not all(
        bool(
            torch.isfinite(
                gradient
            ).all()
        )
        for gradient
        in gradients
    ):
        raise RuntimeError(
            "Non-finite synthetic gradients detected."
        )

    if not bool(
        torch.isfinite(
            logits
        ).all()
    ):
        raise RuntimeError(
            "Non-finite synthetic logits detected."
        )

    if not bool(
        torch.isfinite(
            loss
        )
    ):
        raise RuntimeError(
            "Non-finite synthetic loss detected."
        )

    return {
        "output_shape":
            list(
                logits.shape
            ),

        "parameter_count":
            count_trainable_parameters(
                model
            ),

        "finite_logits":
            True,

        "finite_loss":
            True,

        "finite_gradients":
            True,
    }


def main():

    if OUT_JSON.exists():
        raise FileExistsError(
            OUT_JSON
        )

    if OUT_MD.exists():
        raise FileExistsError(
            OUT_MD
        )

    set_seed(
        42
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Device:",
        device,
    )

    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(
                0
            ),
        )

    report = {
        "identity":
            "deep_learning_architecture_smoke_v2",

        "synthetic_only":
            True,

        "real_research_data_loaded":
            False,

        "research_model_training_performed":
            False,

        "performance_observed":
            False,

        "torch_version":
            torch.__version__,

        "cuda_available":
            torch.cuda.is_available(),

        "device":
            str(device),

        "protocol_sha256":
            sha256(
                PROTOCOL
            ),

        "seed":
            42,

        "shallowconvnet":
            {},

        "multibranch_tcn":
            {},
    }


    print(
        "\n===== SHALLOWCONVNET SYNTHETIC FORWARD/BACKWARD ====="
    )

    for seconds in [
        4,
        8,
        16,
        24,
        32,
    ]:

        n_times = (
            128
            * seconds
        )

        model = ShallowConvNet(
            n_classes=3,
            n_times=n_times,
        )

        x = torch.randn(
            2,
            8,
            n_times,
        )

        result = one_step(
            model,
            x,
            classes=3,
            device=device,
        )

        report[
            "shallowconvnet"
        ][
            f"{seconds}s"
        ] = result

        print(
            f"{seconds:>2}s | "
            f"output={result['output_shape']} | "
            f"parameters={result['parameter_count']:,} | "
            f"finite gradients=True"
        )

        del model
        del x

        if device.type == "cuda":
            torch.cuda.empty_cache()


    print(
        "\n===== MULTIBRANCH TCN SEVEN-PATH SYNTHETIC FORWARD/BACKWARD ====="
    )

    base_inputs = {
        "EEG":
            torch.randn(
                2,
                8,
                512,
            ),

        "ECG":
            torch.randn(
                2,
                1,
                512,
            ),

        "Pupil":
            torch.randn(
                2,
                1,
                120,
            ),
    }

    for path in (
        LOCKED_MODALITY_PATHS
    ):

        model = MultibranchTCN(
            path=path,
            n_classes=3,
        )

        selected_inputs = {
            modality:
                base_inputs[
                    modality
                ]
            for modality
            in LOCKED_MODALITY_PATHS[
                path
            ]
        }

        result = one_step(
            model,
            selected_inputs,
            classes=3,
            device=device,
        )

        report[
            "multibranch_tcn"
        ][
            path
        ] = result

        print(
            f"{path:<14} | "
            f"output={result['output_shape']} | "
            f"parameters={result['parameter_count']:,} | "
            f"finite gradients=True"
        )

        del model

        if device.type == "cuda":
            torch.cuda.empty_cache()


    print(
        "\n===== BBBD BINARY-HEAD SYNTHETIC CHECK ====="
    )

    binary_model = MultibranchTCN(
        path="ECG_EEG_Pupil",
        n_classes=2,
    )

    binary_result = one_step(
        binary_model,
        base_inputs,
        classes=2,
        device=device,
    )

    report[
        "bbbd_binary_all_path"
    ] = binary_result

    print(
        "ECG_EEG_Pupil binary | "
        f"output={binary_result['output_shape']} | "
        f"parameters={binary_result['parameter_count']:,}"
    )


    report[
        "architecture_gate_passed"
    ] = True

    report[
        "next_gate"
    ] = (
        "deep-learning execution-plan generation "
        "and real-data dry-run/preflight"
    )

    report[
        "real_training_allowed_after_this_step"
    ] = False


    OUT_JSON.write_text(
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


    md = [
        "# Deep-Learning Architecture Smoke Test v1",
        "",
        "## Status",
        "",
        "**PASSED**",
        "",
        "This test used synthetic random tensors only. "
        "No participant, BBBD, engineered-feature, or real-label data were loaded.",
        "",
        "## ShallowConvNet",
        "",
        "| Window | Output | Trainable parameters | Finite backward pass |",
        "|---:|---|---:|---|",
    ]

    for seconds in [
        4,
        8,
        16,
        24,
        32,
    ]:

        item = report[
            "shallowconvnet"
        ][
            f"{seconds}s"
        ]

        md.append(
            f"| {seconds} s | "
            f"{item['output_shape']} | "
            f"{item['parameter_count']:,} | yes |"
        )


    md.extend(
        [
            "",
            "## Multibranch TCN",
            "",
            "| Path | Output | Trainable parameters | Finite backward pass |",
            "|---|---|---:|---|",
        ]
    )

    for path in (
        LOCKED_MODALITY_PATHS
    ):

        item = report[
            "multibranch_tcn"
        ][
            path
        ]

        md.append(
            f"| {path} | "
            f"{item['output_shape']} | "
            f"{item['parameter_count']:,} | yes |"
        )


    md.extend(
        [
            "",
            "## Scientific gate",
            "",
            "- Architecture definitions match the frozen protocol.",
            "- ShallowConvNet remains EEG-only.",
            "- Multibranch TCN supports exactly seven locked modality paths.",
            "- Native modality temporal lengths are handled independently before fusion.",
            "- Synthetic logits, losses, and gradients were finite.",
            "- No real-data model fitting occurred.",
            "- No performance metric was observed.",
            "- Real training remains disabled until the execution-plan preflight is frozen.",
        ]
    )


    OUT_MD.write_text(
        "\n".join(
            md
        ).rstrip()
        + "\n",
        encoding="utf-8",
    )


    print(
        "\nARCHITECTURE SMOKE TEST: PASSED"
    )

    print(
        "Real research data loaded: False"
    )

    print(
        "Research-model training performed: False"
    )

    print(
        "Performance observed: False"
    )

    print(
        "Real training allowed now: False"
    )

    print(
        "\nCreated:",
        OUT_JSON.relative_to(
            ROOT
        ),
    )

    print(
        "Created:",
        OUT_MD.relative_to(
            ROOT
        ),
    )


if __name__ == "__main__":
    main()
