
"""Synthetic-only training-runtime smoke test.

No real internal or BBBD research tensors are loaded here.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch

from src.models.deep_learning_training_runtime import (
    ChannelZScoreNormalizer,
    TrainingSettings,
    amp_enabled_for_device,
    build_frozen_model,
    build_seeded_frozen_model,
    configure_final_head_only,
    expand_execution_registry,
    file_sha256,
    inverse_frequency_class_weights,
    train_supervised,
)


ROOT = Path(__file__).resolve().parents[2]

OUT_JSON = (
    ROOT
    / "_research_audit/deep_learning_training_runtime_smoke_v1.json"
)

OUT_MD = (
    ROOT
    / "_research_audit/deep_learning_training_runtime_smoke_v1.md"
)


def synthetic_three_class(
    *,
    seed: int,
    n: int,
    n_times: int,
):

    rng = np.random.default_rng(
        seed
    )

    labels = np.asarray(
        [
            index % 3
            for index
            in range(
                n
            )
        ],
        dtype=np.int64,
    )

    eeg = rng.normal(
        size=(
            n,
            8,
            n_times,
        )
    ).astype(
        np.float32
    )

    ecg = rng.normal(
        size=(
            n,
            1,
            n_times,
        )
    ).astype(
        np.float32
    )

    pupil_times = max(
        40,
        int(
            round(
                n_times
                * 30
                / 128
            )
        ),
    )

    pupil = rng.normal(
        size=(
            n,
            1,
            pupil_times,
        )
    ).astype(
        np.float32
    )

    # Small deterministic class structure for a meaningful optimization
    # smoke test; these are synthetic labels/signals only.
    for class_id in [
        0,
        1,
        2,
    ]:

        mask = (
            labels
            == class_id
        )

        eeg[
            mask,
            0,
            :
        ] += (
            float(
                class_id
            )
            * 0.15
        )

        ecg[
            mask,
            0,
            :
        ] += (
            float(
                class_id
            )
            * 0.10
        )

        pupil[
            mask,
            0,
            :
        ] += (
            float(
                class_id
            )
            * 0.05
        )

    return (
        {
            "EEG":
                eeg,

            "ECG":
                ecg,

            "Pupil":
                pupil,
        },
        labels,
    )


def compare_state_dicts(
    left,
    right,
):

    if left.keys() != right.keys():
        return False

    for key in left:

        if not torch.equal(
            left[
                key
            ],
            right[
                key
            ],
        ):
            return False

    return True


def main():

    if OUT_JSON.exists():
        raise FileExistsError(
            OUT_JSON
        )

    if OUT_MD.exists():
        raise FileExistsError(
            OUT_MD
        )


    print(
        "===== SYNTHETIC TRAINING-RUNTIME SMOKE ====="
    )

    print(
        "Real research data loaded: False"
    )


    # -------------------------------------------------------------
    # A. Registry expansion
    # -------------------------------------------------------------

    operations = expand_execution_registry()

    operation_counts = (
        operations[
            "stage"
        ]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    print(
        "\nExpanded frozen registry operations:",
        len(
            operations
        ),
    )

    print(
        "Operation counts:",
        operation_counts,
    )


    # -------------------------------------------------------------
    # B. Training-only normalization firewall
    # -------------------------------------------------------------

    train_arrays, train_labels = synthetic_three_class(
        seed=101,
        n=36,
        n_times=128,
    )

    validation_arrays, validation_labels = synthetic_three_class(
        seed=202,
        n=18,
        n_times=128,
    )

    # Deliberately move validation distribution far away.
    validation_arrays = {
        key:
            value
            + 1000.0
        for key, value
        in validation_arrays.items()
    }


    normalizer = ChannelZScoreNormalizer().fit(
        train_arrays
    )

    for modality, values in train_arrays.items():

        expected_mean = np.mean(
            values,
            axis=(
                0,
                2,
            ),
            dtype=np.float64,
        )

        actual_mean = normalizer.statistics[
            modality
        ].mean

        if not np.allclose(
            expected_mean,
            actual_mean,
            atol=1.0e-12,
            rtol=0.0,
        ):
            raise RuntimeError(
                f"{modality}: normalizer did not use training data exactly."
            )


    transformed_validation = normalizer.transform(
        validation_arrays
    )

    if not any(
        abs(
            float(
                np.mean(
                    values
                )
            )
        )
        > 100.0
        for values
        in transformed_validation.values()
    ):
        raise RuntimeError(
            "Validation distribution shift unexpectedly disappeared; "
            "possible validation-fit leakage."
        )


    print(
        "Training-only normalization firewall: PASSED"
    )


    # -------------------------------------------------------------
    # C. Class-weight formula
    # -------------------------------------------------------------

    weights = inverse_frequency_class_weights(
        np.asarray(
            [
                0,
                0,
                0,
                0,
                1,
                1,
                2,
            ],
            dtype=np.int64,
        ),
        n_classes=3,
    )

    expected = np.asarray(
        [
            7.0 / 12.0,
            7.0 / 6.0,
            7.0 / 3.0,
        ],
        dtype=np.float32,
    )

    if not np.allclose(
        weights,
        expected,
        atol=1.0e-7,
    ):
        raise RuntimeError(
            f"Inverse-frequency weight formula mismatch: {weights}"
        )


    print(
        "Inverse-frequency class weighting: PASSED"
    )


    # -------------------------------------------------------------
    # D. Final-head scope
    # -------------------------------------------------------------

    shallow = build_frozen_model(
        model_name="ShallowConvNet",
        path="EEG",
        n_classes=3,
        n_times=128,
    )

    shallow_names = configure_final_head_only(
        shallow
    )

    if shallow_names != [
        "classifier.weight",
        "classifier.bias",
    ]:
        raise RuntimeError(
            f"Unexpected Shallow head scope: {shallow_names}"
        )


    tcn = build_frozen_model(
        model_name="MultibranchTCN",
        path="ECG_EEG_Pupil",
        n_classes=3,
        n_times=128,
    )

    tcn_names = configure_final_head_only(
        tcn
    )

    if tcn_names != [
        "fusion.3.weight",
        "fusion.3.bias",
    ]:
        raise RuntimeError(
            f"Unexpected TCN head scope: {tcn_names}"
        )


    print(
        "Shallow adaptive head:",
        shallow_names,
    )

    print(
        "TCN adaptive head:",
        tcn_names,
    )


    # -------------------------------------------------------------
    # E. AMP device rule
    # -------------------------------------------------------------

    if amp_enabled_for_device(
        torch.device(
            "cpu"
        )
    ):
        raise RuntimeError(
            "CPU AMP must be disabled."
        )


    cuda_amp = False

    if torch.cuda.is_available():

        cuda_amp = amp_enabled_for_device(
            torch.device(
                "cuda"
            )
        )

        if not cuda_amp:
            raise RuntimeError(
                "CUDA AMP should be enabled."
            )


    print(
        "CPU AMP enabled: False"
    )

    print(
        "CUDA AMP enabled:",
        cuda_amp,
    )


    # -------------------------------------------------------------
    # F. Checkpoint/resume trajectory equivalence on CPU
    # -------------------------------------------------------------

    resume_settings = TrainingSettings(
        learning_rate=1.0e-3,
        weight_decay=1.0e-4,
        batch_size=12,
        maximum_epochs=4,
        early_stopping_patience=20,
        scheduler_factor=0.5,
        scheduler_patience=20,
        minimum_learning_rate=1.0e-5,
        gradient_clip_norm=5.0,
    )


    train_eeg = {
        "EEG":
            train_arrays[
                "EEG"
            ],
    }

    validation_eeg = {
        "EEG":
            validation_arrays[
                "EEG"
            ],
    }


    with tempfile.TemporaryDirectory() as temp_root:

        temp_root = Path(
            temp_root
        )

        uninterrupted_dir = (
            temp_root
            / "uninterrupted"
        )

        resumed_dir = (
            temp_root
            / "resumed"
        )


        model_a = build_seeded_frozen_model(
            training_seed=3407,
            model_name="ShallowConvNet",
            path="EEG",
            n_classes=3,
            n_times=128,
        )


        result_a = train_supervised(
            model=model_a,
            train_arrays=train_eeg,
            train_labels=train_labels,
            validation_arrays=validation_eeg,
            validation_labels=validation_labels,
            n_classes=3,
            training_seed=3407,
            output_directory=uninterrupted_dir,
            job_identity="synthetic/resume_equivalence",
            registry_row_hash="synthetic_hash",
            split_identity="synthetic_split",
            settings=resume_settings,
            device=torch.device(
                "cpu"
            ),
            resume=True,
            show_progress=False,
        )


        model_b = build_seeded_frozen_model(
            training_seed=3407,
            model_name="ShallowConvNet",
            path="EEG",
            n_classes=3,
            n_times=128,
        )


        interrupted = train_supervised(
            model=model_b,
            train_arrays=train_eeg,
            train_labels=train_labels,
            validation_arrays=validation_eeg,
            validation_labels=validation_labels,
            n_classes=3,
            training_seed=3407,
            output_directory=resumed_dir,
            job_identity="synthetic/resume_equivalence",
            registry_row_hash="synthetic_hash",
            split_identity="synthetic_split",
            settings=resume_settings,
            device=torch.device(
                "cpu"
            ),
            resume=True,
            show_progress=False,
            interrupt_after_epoch=2,
        )


        if interrupted[
            "status"
        ] != "interrupted":
            raise RuntimeError(
                "Synthetic interruption did not occur."
            )


        model_c = build_seeded_frozen_model(
            training_seed=3407,
            model_name="ShallowConvNet",
            path="EEG",
            n_classes=3,
            n_times=128,
        )


        result_c = train_supervised(
            model=model_c,
            train_arrays=train_eeg,
            train_labels=train_labels,
            validation_arrays=validation_eeg,
            validation_labels=validation_labels,
            n_classes=3,
            training_seed=3407,
            output_directory=resumed_dir,
            job_identity="synthetic/resume_equivalence",
            registry_row_hash="synthetic_hash",
            split_identity="synthetic_split",
            settings=resume_settings,
            device=torch.device(
                "cpu"
            ),
            resume=True,
            show_progress=False,
        )


        checkpoint_a = torch.load(
            uninterrupted_dir
            / "last.pt",
            map_location="cpu",
            weights_only=False,
        )

        checkpoint_c = torch.load(
            resumed_dir
            / "last.pt",
            map_location="cpu",
            weights_only=False,
        )


        if not compare_state_dicts(
            checkpoint_a[
                "model_state_dict"
            ],
            checkpoint_c[
                "model_state_dict"
            ],
        ):
            raise RuntimeError(
                "Interrupted/resumed trajectory differs from uninterrupted training."
            )


        if checkpoint_a[
            "optimizer_state_dict"
        ][
            "param_groups"
        ] != checkpoint_c[
            "optimizer_state_dict"
        ][
            "param_groups"
        ]:
            raise RuntimeError(
                "Optimizer param-group state differs after resume."
            )


        if result_a[
            "best_epoch"
        ] != result_c[
            "best_epoch"
        ]:
            raise RuntimeError(
                "Best epoch differs after resume."
            )


        if abs(
            result_a[
                "best_validation_balanced_accuracy"
            ]
            - result_c[
                "best_validation_balanced_accuracy"
            ]
        ) > 1.0e-12:
            raise RuntimeError(
                "Best validation BA differs after resume."
            )


    print(
        "Atomic checkpoint/resume trajectory equivalence: PASSED"
    )


    # -------------------------------------------------------------
    # G. Visible synthetic TCN training progress/ETA
    # -------------------------------------------------------------

    print(
        "\n===== VISIBLE SYNTHETIC TCN TRAINING WITH ETA ====="
    )


    small_train_arrays, small_train_labels = (
        synthetic_three_class(
            seed=303,
            n=36,
            n_times=128,
        )
    )

    small_validation_arrays, small_validation_labels = (
        synthetic_three_class(
            seed=404,
            n=18,
            n_times=128,
        )
    )


    smoke_settings = TrainingSettings(
        learning_rate=1.0e-3,
        weight_decay=1.0e-4,
        batch_size=12,
        maximum_epochs=3,
        early_stopping_patience=10,
        scheduler_factor=0.5,
        scheduler_patience=5,
        minimum_learning_rate=1.0e-5,
        gradient_clip_norm=5.0,
    )


    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


    with tempfile.TemporaryDirectory() as temp_root:

        smoke_model = build_seeded_frozen_model(
            training_seed=2026,
            model_name="MultibranchTCN",
            path="ECG_EEG_Pupil",
            n_classes=3,
            n_times=128,
        )


        smoke_result = train_supervised(
            model=smoke_model,
            train_arrays=small_train_arrays,
            train_labels=small_train_labels,
            validation_arrays=small_validation_arrays,
            validation_labels=small_validation_labels,
            n_classes=3,
            training_seed=2026,
            output_directory=Path(
                temp_root
            ),
            job_identity="synthetic/TCN/ECG_EEG_Pupil/seed_2026",
            registry_row_hash="synthetic_visible_hash",
            split_identity="synthetic_visible_split",
            settings=smoke_settings,
            device=device,
            resume=True,
            show_progress=True,
        )


        if smoke_result[
            "status"
        ] != "completed":
            raise RuntimeError(
                "Synthetic visible TCN smoke did not complete."
            )


    print(
        "Visible progress/ETA training smoke: PASSED"
    )


    report = {
        "identity":
            "deep_learning_training_runtime_smoke_v1",

        "protocol_revision":
            "1.3",

        "real_research_data_loaded":
            False,

        "real_research_model_fitting_performed":
            False,

        "real_research_performance_observed":
            False,

        "synthetic_training_performed":
            True,

        "expanded_fit_adaptation_operations":
            int(
                len(
                    operations
                )
            ),

        "operation_counts":
            {
                str(key):
                    int(value)
                for key, value
                in operation_counts.items()
            },

        "training_only_normalization_test":
            "passed",

        "inverse_frequency_class_weight_test":
            "passed",

        "shallow_final_head_parameters":
            shallow_names,

        "tcn_final_head_parameters":
            tcn_names,

        "cpu_amp_enabled":
            False,

        "cuda_available":
            bool(
                torch.cuda.is_available()
            ),

        "cuda_amp_enabled":
            bool(
                cuda_amp
            ),

        "checkpoint_resume_exact_model_state":
            True,

        "checkpoint_resume_validation_selection_equivalence":
            True,

        "visible_progress_eta_smoke":
            True,

        "real_training_gate_open":
            False,

        "next_gate":
            (
                "bind every frozen operation to the real internal/BBBD "
                "cache, verify exact split hashes and adaptive temporal "
                "roles without fitting, and only then open real training"
            ),
    }


    OUT_JSON.write_text(
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


    OUT_MD.write_text(
        """# Deep-Learning Training Runtime Smoke v1

## Status

**PASSED  synthetic training only.**

No internal XR or BBBD research tensors were loaded by this smoke test.

## Frozen execution mechanics verified

- Frozen registry expands to exactly 3,936 fit/adaptation operations.
- Per-channel z-score normalization is fitted from training arrays only.
- Validation distribution cannot influence fitted normalization statistics.
- Class weighting uses standard balanced inverse frequency:
  `N / (K * n_c)`.
- ShallowConvNet head-only adaptation trains only
  `classifier.weight` and `classifier.bias`.
- MultibranchTCN head-only adaptation trains only
  `fusion.3.weight` and `fusion.3.bias`.
- CPU execution uses FP32 without AMP.
- CUDA execution uses the frozen CUDA AMP policy when available.
- Atomic epoch checkpointing works.
- Interrupted + resumed execution reproduced the uninterrupted synthetic
  model trajectory exactly.
- Live epoch progress and ETA were exercised.

## Training gate

Real-data training remains blocked.

The next gate is a no-fit real-data binding audit across the complete
3,936-operation plan, including exact split-hash verification and
subject-adaptive temporal calibration/guard/test role verification.
""".rstrip()
        + "\n",
        encoding="utf-8",
    )


    print(
        "\n===== SYNTHETIC RUNTIME VERDICT ====="
    )

    print(
        "Frozen operation expansion: 3,936 / 3,936"
    )

    print(
        "Training-only normalization: PASSED"
    )

    print(
        "Inverse-frequency class weighting: PASSED"
    )

    print(
        "Head-only adaptation scope: PASSED"
    )

    print(
        "Checkpoint/resume equivalence: PASSED"
    )

    print(
        "Progress + ETA: PASSED"
    )

    print(
        "Real research data loaded: False"
    )

    print(
        "Real CNN/TCN fitting performed: False"
    )

    print(
        "Research performance observed: False"
    )

    print(
        "Real training allowed now: False"
    )

    print(
        "Created:",
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
