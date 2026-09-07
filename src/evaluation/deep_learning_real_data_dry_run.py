
"""Forward-only real-data dry-run for frozen deep baselines.

This script loads genuine internal and BBBD signals and verifies their
compatibility with the frozen neural architectures.

It deliberately:
- performs no optimizer step;
- computes no loss;
- compares no prediction with a label;
- computes no accuracy, F1, AUC, or other performance metric.
"""

from __future__ import annotations

import json
import tempfile
import zipfile
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import torch
import yaml

from src.core.config import load_revision_config
from src.features.bbbd_features import (
    derivative_eeg_member,
    extract_recording,
    load_bbbd_config,
    preprocess_ecg,
    preprocess_pupil,
    read_headerless_tsv_gz,
)

from src.models.deep_learning_baselines import (
    LOCKED_MODALITY_PATHS,
    MultibranchTCN,
    ShallowConvNet,
)


ROOT = Path(__file__).resolve().parents[2]

INTERNAL_FILE = (
    ROOT
    / "outputs/revision/preprocessed/"
      "internal_xr/recordings/P01_phase1.npz"
)

BBBD_MANIFEST = (
    ROOT
    / "artifacts/revision/manifests/"
      "bbbd_experiment2_records.csv"
)

BBBD_RECORDING_AUDIT = (
    ROOT
    / "outputs/revision/features/"
      "bbbd_primary/recording_audit.csv"
)

PROTOCOL_PATH = (
    ROOT
    / "configs/deep_learning_baselines.yaml"
)

OUT_JSON = (
    ROOT
    / "_research_audit/deep_learning_real_data_dry_run_v1.json"
)

OUT_MD = (
    ROOT
    / "_research_audit/deep_learning_real_data_dry_run_v1.md"
)


def forward_shapes(
    *,
    eeg: np.ndarray,
    ecg: np.ndarray,
    pupil: np.ndarray,
    n_classes: int,
    device: torch.device,
) -> dict:

    eeg_tensor = torch.as_tensor(
        eeg,
        dtype=torch.float32,
    ).unsqueeze(
        0
    ).to(
        device
    )

    ecg_tensor = torch.as_tensor(
        ecg,
        dtype=torch.float32,
    ).reshape(
        1,
        1,
        -1,
    ).to(
        device
    )

    pupil_tensor = torch.as_tensor(
        pupil,
        dtype=torch.float32,
    ).reshape(
        1,
        1,
        -1,
    ).to(
        device
    )

    inputs = {
        "EEG":
            eeg_tensor,

        "ECG":
            ecg_tensor,

        "Pupil":
            pupil_tensor,
    }

    report = {}


    shallow = ShallowConvNet(
        n_classes=n_classes,
        n_times=eeg.shape[-1],
    ).to(
        device
    )

    shallow.eval()

    with torch.no_grad():
        logits = shallow(
            eeg_tensor
        )

    if not bool(
        torch.isfinite(
            logits
        ).all()
    ):
        raise RuntimeError(
            "Non-finite ShallowConvNet real-data forward output."
        )

    report[
        "ShallowConvNet_EEG"
    ] = {
        "input_shape":
            list(
                eeg_tensor.shape
            ),

        "output_shape":
            list(
                logits.shape
            ),

        "finite":
            True,
    }


    for path in LOCKED_MODALITY_PATHS:

        model = MultibranchTCN(
            path=path,
            n_classes=n_classes,
        ).to(
            device
        )

        model.eval()

        selected = {
            modality:
                inputs[
                    modality
                ]
            for modality
            in LOCKED_MODALITY_PATHS[
                path
            ]
        }

        with torch.no_grad():
            logits = model(
                selected
            )

        if not bool(
            torch.isfinite(
                logits
            ).all()
        ):
            raise RuntimeError(
                f"Non-finite TCN output for {path}."
            )

        report[
            f"MultibranchTCN_{path}"
        ] = {
            "output_shape":
                list(
                    logits.shape
                ),

            "finite":
                True,
        }

    return report


def internal_dry_run(
    device: torch.device,
):

    if not INTERNAL_FILE.is_file():
        raise FileNotFoundError(
            INTERNAL_FILE
        )

    with np.load(
        INTERNAL_FILE,
        allow_pickle=False,
    ) as archive:

        eeg = archive[
            "eeg"
        ]

        ecg = archive[
            "ecg"
        ]

        pupil = archive[
            "pupil"
        ]

        pupil_available = archive[
            "pupil_available"
        ].astype(bool)

        pupil_columns = [
            str(value)
            for value
            in archive[
                "pupil_columns"
            ].tolist()
        ]


        if pupil_columns != [
            "x",
            "y",
            "r",
        ]:
            raise RuntimeError(
                f"Unexpected internal pupil columns: {pupil_columns}"
            )

        valid_indices = np.flatnonzero(
            pupil_available
        )

        if len(
            valid_indices
        ) == 0:
            raise RuntimeError(
                "Representative internal file has no valid pupil window."
            )

        index = int(
            valid_indices[
                0
            ]
        )

        eeg_window = np.asarray(
            eeg[
                index
            ],
            dtype=np.float32,
        ).T

        ecg_window = np.asarray(
            ecg[
                index
            ],
            dtype=np.float32,
        )

        # Frozen protocol revision 1.1:
        # internal Pupil model input is r only.
        pupil_r = np.asarray(
            pupil[
                index,
                :,
                2,
            ],
            dtype=np.float32,
        )


    if eeg_window.shape != (
        8,
        512,
    ):
        raise RuntimeError(
            f"Unexpected internal EEG shape: {eeg_window.shape}"
        )

    if ecg_window.shape != (
        512,
    ):
        raise RuntimeError(
            f"Unexpected internal ECG shape: {ecg_window.shape}"
        )

    if pupil_r.shape != (
        120,
    ):
        raise RuntimeError(
            f"Unexpected internal pupil-r shape: {pupil_r.shape}"
        )

    if not np.isfinite(
        eeg_window
    ).all():
        raise RuntimeError(
            "Internal EEG window is non-finite."
        )

    if not np.isfinite(
        ecg_window
    ).all():
        raise RuntimeError(
            "Internal ECG window is non-finite."
        )

    if not np.isfinite(
        pupil_r
    ).all():
        raise RuntimeError(
            "Internal pupil-r window is non-finite."
        )


    forwards = forward_shapes(
        eeg=
            eeg_window,

        ecg=
            ecg_window,

        pupil=
            pupil_r,

        n_classes=
            3,

        device=
            device,
    )


    return {
        "source":
            str(
                INTERNAL_FILE.relative_to(
                    ROOT
                )
            ),

        "representative_window_index":
            index,

        "eeg_shape":
            list(
                eeg_window.shape
            ),

        "ecg_shape":
            list(
                ecg_window.shape
            ),

        "pupil_model_input":
            "r",

        "pupil_shape":
            list(
                pupil_r.shape
            ),

        "label_used_for_performance":
            False,

        "forward_checks":
            forwards,
    }


def bbbd_dry_run(
    device: torch.device,
    bbbd_archive: Path,
):

    if not bbbd_archive.is_file():
        raise FileNotFoundError(
            bbbd_archive
        )

    records = pd.read_csv(
        BBBD_MANIFEST,
        low_memory=False,
    )

    selected = records.loc[
        (
            records[
                "subject"
            ].astype(str)
            == "sub-01"
        )
        & (
            records[
                "session"
            ].astype(str)
            == "ses-01"
        )
        & (
            records[
                "task"
            ].astype(str)
            == "stim01"
        )
        & (
            records[
                "primary_cohort"
            ].astype(bool)
        )
    ]

    if len(
        selected
    ) != 1:
        raise RuntimeError(
            f"Expected one representative BBBD record, found {len(selected)}."
        )

    record = selected.iloc[
        0
    ]

    config = load_bbbd_config(
        "configs/bbbd.yaml"
    )

    signal_config = config[
        "signals"
    ]

    fs = float(
        signal_config[
            "sampling_frequency_hz"
        ]
    )

    if fs != 128.0:
        raise RuntimeError(
            f"Unexpected BBBD sampling frequency: {fs}"
        )


    with zipfile.ZipFile(
        bbbd_archive,
        "r",
    ) as archive:

        with tempfile.TemporaryDirectory() as temp_dir:

            extracted, extraction_audit = extract_recording(
                archive,
                record,
                dataset_name=
                    "experiment2",
                config=
                    config,
                temporary_directory=
                    Path(
                        temp_dir
                    ),
            )


            all_path_rows = extracted[
                "ECG_EEG_Pupil"
            ]

            if not all_path_rows:
                raise RuntimeError(
                    "Representative BBBD recording produced no accepted windows."
                )

            first_accepted = all_path_rows[
                0
            ]

            start_seconds = float(
                first_accepted[
                    "start_seconds"
                ]
            )

            end_seconds = float(
                first_accepted[
                    "end_seconds"
                ]
            )


            eeg_member = derivative_eeg_member(
                str(
                    record[
                        "eeg_path"
                    ]
                )
            )

            bdf_path = (
                Path(
                    temp_dir
                )
                / "representative_desc-eeg.bdf"
            )

            bdf_path.write_bytes(
                archive.read(
                    eeg_member
                )
            )

            raw = mne.io.read_raw_bdf(
                bdf_path,
                preload=True,
                verbose="ERROR",
            )

            eeg_channels = [
                str(value)
                for value
                in signal_config[
                    "eeg"
                ][
                    "channels"
                ]
            ]

            if any(
                channel not in raw.ch_names
                for channel
                in eeg_channels
            ):
                raise RuntimeError(
                    "Required BBBD EEG channel missing."
                )

            eeg_full = raw.get_data(
                picks=
                    eeg_channels
            )

            if signal_config[
                "eeg"
            ][
                "convert_volts_to_microvolts"
            ]:
                eeg_full = (
                    eeg_full
                    * 1.0e6
                )

            del raw


            ecg_raw = read_headerless_tsv_gz(
                archive,
                str(
                    record[
                        "ecg_path"
                    ]
                ),
                "rawECG",
            )

            ecg_full = preprocess_ecg(
                ecg_raw,
                sampling_frequency_hz=
                    fs,
                bandpass_hz=
                    signal_config[
                        "ecg"
                    ][
                        "bandpass_hz"
                    ],
                filter_order=
                    int(
                        signal_config[
                            "ecg"
                        ][
                            "filter_order"
                        ]
                    ),
                minimum_finite_fraction=
                    float(
                        signal_config[
                            "ecg"
                        ][
                            "minimum_finite_fraction"
                        ]
                    ),
            )


            pupil_raw = read_headerless_tsv_gz(
                archive,
                str(
                    record[
                        "pupil_path"
                    ]
                ),
                "pupil_size",
            )

            maximum_gap_samples = int(
                round(
                    float(
                        signal_config[
                            "pupil"
                        ][
                            "maximum_interpolation_gap_seconds"
                        ]
                    )
                    * fs
                )
            )

            pupil_full, pupil_original_valid = (
                preprocess_pupil(
                    pupil_raw,
                    minimum_valid_value_exclusive=
                        float(
                            signal_config[
                                "pupil"
                            ][
                                "minimum_valid_value_exclusive"
                            ]
                        ),
                    maximum_gap_samples=
                        maximum_gap_samples,
                )
            )


            start = int(
                round(
                    start_seconds
                    * fs
                )
            )

            stop = int(
                round(
                    end_seconds
                    * fs
                )
            )

            expected_window_samples = int(
                round(
                    4.0
                    * fs
                )
            )

            if (
                stop
                - start
                != expected_window_samples
            ):
                raise RuntimeError(
                    "Representative BBBD accepted window is not exactly 4 seconds."
                )


            eeg_window = np.asarray(
                eeg_full[
                    :,
                    start:stop,
                ],
                dtype=np.float32,
            )

            ecg_window = np.asarray(
                ecg_full[
                    start:stop
                ],
                dtype=np.float32,
            )

            pupil_window = np.asarray(
                pupil_full[
                    start:stop
                ],
                dtype=np.float32,
            )

            original_valid = np.asarray(
                pupil_original_valid[
                    start:stop
                ],
                dtype=bool,
            )


    if eeg_window.shape != (
        8,
        512,
    ):
        raise RuntimeError(
            f"Unexpected BBBD EEG shape: {eeg_window.shape}"
        )

    if ecg_window.shape != (
        512,
    ):
        raise RuntimeError(
            f"Unexpected BBBD ECG shape: {ecg_window.shape}"
        )

    if pupil_window.shape != (
        512,
    ):
        raise RuntimeError(
            f"Unexpected BBBD pupil shape: {pupil_window.shape}"
        )

    if not np.isfinite(
        eeg_window
    ).all():
        raise RuntimeError(
            "BBBD EEG window is non-finite."
        )

    if not np.isfinite(
        ecg_window
    ).all():
        raise RuntimeError(
            "BBBD ECG window is non-finite."
        )

    if not np.isfinite(
        pupil_window
    ).all():
        raise RuntimeError(
            "BBBD accepted pupil window is non-finite."
        )


    pupil_valid_fraction = float(
        np.mean(
            original_valid
        )
    )

    if pupil_valid_fraction < 0.8:
        raise RuntimeError(
            "Representative BBBD pupil-valid fraction is below 0.8."
        )


    audit = pd.read_csv(
        BBBD_RECORDING_AUDIT,
        low_memory=False,
    )

    expected = audit.loc[
        audit[
            "recording_id"
        ].astype(str)
        == "experiment2::sub-01::ses-01::stim01"
    ]

    if len(
        expected
    ) != 1:
        raise RuntimeError(
            "Could not resolve representative BBBD recording audit."
        )

    expected = expected.iloc[
        0
    ]

    candidate_windows = int(
        extraction_audit[
            "candidate_windows"
        ]
    )

    accepted_windows = int(
        extraction_audit[
            "accepted_windows"
        ]
    )

    if candidate_windows != int(
        expected[
            "candidate_windows"
        ]
    ):
        raise RuntimeError(
            "BBBD candidate-window parity failed."
        )

    if accepted_windows != int(
        expected[
            "accepted_windows"
        ]
    ):
        raise RuntimeError(
            "BBBD accepted-window parity failed."
        )


    forwards = forward_shapes(
        eeg=
            eeg_window,

        ecg=
            ecg_window,

        pupil=
            pupil_window,

        n_classes=
            2,

        device=
            device,
    )


    return {
        "recording_id":
            "experiment2::sub-01::ses-01::stim01",

        "eeg_shape":
            list(
                eeg_window.shape
            ),

        "ecg_shape":
            list(
                ecg_window.shape
            ),

        "pupil_model_input":
            "pupil_size",

        "pupil_shape":
            list(
                pupil_window.shape
            ),

        "pupil_valid_fraction":
            pupil_valid_fraction,

        "candidate_windows":
            candidate_windows,

        "accepted_windows":
            accepted_windows,

        "locked_candidate_window_parity":
            True,

        "locked_accepted_window_parity":
            True,

        "label_used_for_performance":
            False,

        "forward_checks":
            forwards,
    }


def main():

    project_config = load_revision_config(
        check_input_paths=False
    )

    bbbd_archive = Path(
        project_config["paths"]["bbbd_experiment2_archive"]
    )

    if not bbbd_archive.is_file():
        raise FileNotFoundError(
            f"Configured BBBD Experiment 2 archive "
            f"does not exist: {bbbd_archive}"
        )

    if OUT_JSON.exists():
        raise FileExistsError(
            OUT_JSON
        )

    if OUT_MD.exists():
        raise FileExistsError(
            OUT_MD
        )

    protocol = yaml.safe_load(
        PROTOCOL_PATH.read_text(
            encoding="utf-8"
        )
    )

    if protocol[
        "protocol_revision"
    ] != "1.1":
        raise RuntimeError(
            "Expected protocol revision 1.1."
        )

    if protocol[
        "models"
    ][
        "multibranch_tcn"
    ][
        "branch_input_channels"
    ][
        "Pupil"
    ] != 1:
        raise RuntimeError(
            "Expected one-channel Pupil TCN branch."
        )


    torch.manual_seed(
        42
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
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


    print(
        "\n===== INTERNAL REAL-DATA FORWARD-ONLY CHECK ====="
    )

    internal = internal_dry_run(
        device
    )

    print(
        "Internal EEG:",
        internal[
            "eeg_shape"
        ],
    )

    print(
        "Internal ECG:",
        internal[
            "ecg_shape"
        ],
    )

    print(
        "Internal pupil:",
        internal[
            "pupil_shape"
        ],
        "(r only)",
    )

    print(
        "Internal model/path forward checks:",
        len(
            internal[
                "forward_checks"
            ]
        ),
        "/ 8 passed",
    )


    print(
        "\n===== BBBD REAL-DATA FORWARD-ONLY CHECK ====="
    )

    bbbd = bbbd_dry_run(
        device,
        bbbd_archive,
    )

    print(
        "BBBD recording:",
        bbbd[
            "recording_id"
        ],
    )

    print(
        "BBBD EEG:",
        bbbd[
            "eeg_shape"
        ],
    )

    print(
        "BBBD ECG:",
        bbbd[
            "ecg_shape"
        ],
    )

    print(
        "BBBD pupil:",
        bbbd[
            "pupil_shape"
        ],
        "(pupil_size only)",
    )

    print(
        "Candidate windows:",
        bbbd[
            "candidate_windows"
        ],
    )

    print(
        "Accepted windows:",
        bbbd[
            "accepted_windows"
        ],
    )

    print(
        "BBBD model/path forward checks:",
        len(
            bbbd[
                "forward_checks"
            ]
        ),
        "/ 8 passed",
    )


    report = {
        "identity":
            "deep_learning_real_data_dry_run_v1",

        "protocol_revision":
            "1.1",

        "device":
            str(
                device
            ),

        "torch_version":
            torch.__version__,

        "real_research_data_loaded":
            True,

        "optimizer_created":
            False,

        "loss_computed":
            False,

        "backward_pass_performed":
            False,

        "parameter_update_performed":
            False,

        "labels_compared_with_predictions":
            False,

        "performance_metrics_computed":
            False,

        "performance_observed":
            False,

        "internal":
            internal,

        "bbbd":
            bbbd,

        "real_data_forward_only_gate_passed":
            True,

        "next_gate":
            (
                "full deterministic deep-learning raw-input cache "
                "construction and complete-universe parity audit"
            ),

        "real_training_allowed_after_this_step":
            False,
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


    lines = [
        "# Deep-Learning Real-Data Dry Run v1",
        "",
        "## Status",
        "",
        "**PASSED  forward only.**",
        "",
        "Real internal and BBBD signals were loaded, but no loss, backward pass, optimizer step, or performance metric was computed.",
        "",
        "## Internal",
        "",
        f"- EEG input: {internal['eeg_shape']}.",
        f"- ECG input: {internal['ecg_shape']}.",
        f"- Pupil input: {internal['pupil_shape']} using `r` only.",
        "- ShallowConvNet EEG forward: passed.",
        "- Seven TCN modality-path forwards: passed.",
        "",
        "## BBBD",
        "",
        f"- Recording: `{bbbd['recording_id']}`.",
        f"- EEG input: {bbbd['eeg_shape']}.",
        f"- ECG input: {bbbd['ecg_shape']}.",
        f"- Pupil input: {bbbd['pupil_shape']} using `pupil_size` only.",
        f"- Candidate-window parity: {bbbd['candidate_windows']}.",
        f"- Accepted-window parity: {bbbd['accepted_windows']}.",
        "- ShallowConvNet EEG forward: passed.",
        "- Seven TCN modality-path forwards: passed.",
        "",
        "## Performance firewall",
        "",
        "- Optimizer created: no.",
        "- Loss computed: no.",
        "- Backward pass: no.",
        "- Parameter update: no.",
        "- Predictions compared with labels: no.",
        "- Performance metrics computed: no.",
        "",
        "## Next gate",
        "",
        "Build the complete deterministic raw-input cache and prove parity with the full locked internal and BBBD window universes before allowing real model fitting.",
    ]

    OUT_MD.write_text(
        "\n".join(
            lines
        ).rstrip()
        + "\n",
        encoding="utf-8",
    )


    print(
        "\nREAL-DATA FORWARD-ONLY DRY RUN: PASSED"
    )

    print(
        "Optimizer created: False"
    )

    print(
        "Loss computed: False"
    )

    print(
        "Backward pass performed: False"
    )

    print(
        "Performance metrics computed: False"
    )

    print(
        "Performance observed: False"
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
