
"""Deterministic raw-signal cache for frozen deep-learning baselines.

This module prepares neural-network inputs only.

It deliberately performs:
- no training-set normalization fitting;
- no optimizer construction;
- no loss computation;
- no backward pass;
- no neural-network fitting;
- no prediction/performance evaluation.

The cache preserves raw/preprocessed signal values already locked by the
internal and BBBD preprocessing contracts.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import tempfile
import time
import zipfile
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import yaml

from src.core.config import load_revision_config
from src.features.bbbd_features import (
    derivative_eeg_member,
    load_bbbd_config,
    preprocess_ecg,
    preprocess_pupil,
    read_headerless_tsv_gz,
)


ROOT = Path(__file__).resolve().parents[2]

PROTOCOL = (
    ROOT
    / "configs/deep_learning_baselines.yaml"
)

INTERNAL_ROOT = (
    ROOT
    / "outputs/revision/preprocessed/internal_xr/recordings"
)

INTERNAL_MANIFEST = (
    ROOT
    / "outputs/revision/preprocessed/internal_xr/window_manifest.csv"
)

BBBD_FEATURE_TABLE = (
    ROOT
    / "outputs/revision/features/"
      "bbbd_primary/ECG_EEG_Pupil_features.csv"
)

BBBD_RECORDING_AUDIT = (
    ROOT
    / "outputs/revision/features/"
      "bbbd_primary/recording_audit.csv"
)

BBBD_EXP2_MANIFEST = (
    ROOT
    / "artifacts/revision/manifests/"
      "bbbd_experiment2_records.csv"
)

BBBD_EXP3_MANIFEST = (
    ROOT
    / "artifacts/revision/manifests/"
      "bbbd_experiment3_records.csv"
)

BBBD_COHORT = (
    ROOT
    / "artifacts/revision/manifests/"
      "bbbd_multimodal_cohort_registry.csv"
)

CACHE_ROOT = (
    ROOT
    / "outputs/revision/deep_learning_cache_v1"
)

CACHE_INTERNAL = (
    CACHE_ROOT
    / "internal_xr"
)

CACHE_BBBD = (
    CACHE_ROOT
    / "bbbd"
)

AUDIT_JSON = (
    ROOT
    / "_research_audit/deep_learning_raw_cache_audit_v1.json"
)

AUDIT_MD = (
    ROOT
    / "_research_audit/deep_learning_raw_cache_audit_v1.md"
)

CACHE_MANIFEST = (
    ROOT
    / "_research_audit/deep_learning_raw_cache_manifest_v1.csv"
)


EEG_CHANNELS = [
    "AF7",
    "Fp1",
    "Fpz",
    "Fp2",
    "AF8",
    "O1",
    "POz",
    "O2",
]


def sha256_file(
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


class Progress:

    def __init__(
        self,
        total: int,
        description: str,
    ) -> None:

        self.total = int(
            total
        )

        self.description = str(
            description
        )

        self.start = time.time()
        self.count = 0

        try:
            from tqdm import tqdm

            self.bar = tqdm(
                total=self.total,
                desc=self.description,
                unit="recording",
                dynamic_ncols=True,
            )

        except Exception:
            self.bar = None

            print(
                f"{self.description}: 0/{self.total}"
            )

    def update(
        self,
        context: str = "",
    ) -> None:

        self.count += 1

        if self.bar is not None:

            if context:
                self.bar.set_postfix_str(
                    context
                )

            self.bar.update(
                1
            )

            return

        elapsed = max(
            time.time()
            - self.start,
            1.0e-9,
        )

        rate = (
            self.count
            / elapsed
        )

        remaining = (
            (
                self.total
                - self.count
            )
            / rate
            if rate > 0
            else float("inf")
        )

        print(
            f"{self.description}: "
            f"{self.count}/{self.total} | "
            f"elapsed={elapsed / 60:.1f} min | "
            f"ETA={remaining / 60:.1f} min | "
            f"{context}",
            flush=True,
        )

    def close(
        self,
    ) -> None:

        if self.bar is not None:
            self.bar.close()


def ensure_clean_target() -> None:

    for path in [
        AUDIT_JSON,
        AUDIT_MD,
        CACHE_MANIFEST,
    ]:
        if path.exists():
            raise FileExistsError(
                f"Audit output already exists: {path}"
            )

    if CACHE_ROOT.exists():
        raise FileExistsError(
            f"Cache root already exists: {CACHE_ROOT}"
        )

    usage = shutil.disk_usage(
        CACHE_ROOT.parent
    )

    free_gb = (
        usage.free
        / 1024**3
    )

    print(
        f"Free disk space: {free_gb:.2f} GB"
    )

    if free_gb < 8.0:
        raise RuntimeError(
            "At least 8 GB free disk space is required "
            "for conservative cache construction."
        )


def load_protocol() -> dict:

    protocol = yaml.safe_load(
        PROTOCOL.read_text(
            encoding="utf-8"
        )
    )

    if protocol[
        "protocol_revision"
    ] != "1.2":
        raise RuntimeError(
            "Expected protocol revision 1.2."
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
            "Expected one-channel Pupil model input."
        )

    return protocol


def build_internal_cache():
    print(
        "\n===== INTERNAL COMPLETE CACHE ====="
    )

    CACHE_INTERNAL.mkdir(
        parents=True,
        exist_ok=False,
    )

    manifest = pd.read_csv(
        INTERNAL_MANIFEST,
        low_memory=False,
    )

    if len(
        manifest
    ) != 3315:
        raise RuntimeError(
            "Expected 3,315 internal base windows."
        )

    archive_paths = sorted(
        INTERNAL_ROOT.glob(
            "P*_phase*.npz"
        )
    )

    if len(
        archive_paths
    ) != 39:
        raise RuntimeError(
            f"Expected 39 internal archives, found {len(archive_paths)}."
        )

    base_eeg = []
    base_ecg = []
    base_pupil = []
    base_pupil_valid = []
    base_labels = []
    base_participants = []
    base_phases = []
    base_source_archive = []
    base_source_index = []

    archive_sizes = {}

    for path in archive_paths:

        with np.load(
            path,
            allow_pickle=False,
        ) as source:

            channels = [
                str(value)
                for value
                in source[
                    "eeg_channels"
                ].tolist()
            ]

            pupil_columns = [
                str(value)
                for value
                in source[
                    "pupil_columns"
                ].tolist()
            ]

            if channels != EEG_CHANNELS:
                raise RuntimeError(
                    f"{path.name}: EEG channel mismatch."
                )

            if pupil_columns != [
                "x",
                "y",
                "r",
            ]:
                raise RuntimeError(
                    f"{path.name}: pupil source-column mismatch."
                )

            eeg = np.asarray(
                source[
                    "eeg"
                ],
                dtype=np.float32,
            )

            ecg = np.asarray(
                source[
                    "ecg"
                ],
                dtype=np.float32,
            )

            pupil = np.asarray(
                source[
                    "pupil"
                ][
                    :,
                    :,
                    2,
                ],
                dtype=np.float32,
            )

            pupil_valid = source[
                "pupil_available"
            ].astype(
                bool
            )

            label = int(
                np.asarray(
                    source[
                        "label"
                    ]
                ).item()
            )

            phase = int(
                np.asarray(
                    source[
                        "phase"
                    ]
                ).item()
            )

        participant = path.name.split(
            "_",
            1,
        )[0]

        n = int(
            eeg.shape[
                0
            ]
        )

        if eeg.shape != (
            n,
            512,
            8,
        ):
            raise RuntimeError(
                f"{path.name}: EEG source shape mismatch."
            )

        if ecg.shape != (
            n,
            512,
        ):
            raise RuntimeError(
                f"{path.name}: ECG source shape mismatch."
            )

        if pupil.shape != (
            n,
            120,
        ):
            raise RuntimeError(
                f"{path.name}: pupil-r shape mismatch."
            )

        base_eeg.append(
            np.transpose(
                eeg,
                (
                    0,
                    2,
                    1,
                ),
            )
        )

        base_ecg.append(
            ecg[
                :,
                None,
                :,
            ]
        )

        base_pupil.append(
            pupil[
                :,
                None,
                :,
            ]
        )

        base_pupil_valid.append(
            pupil_valid
        )

        base_labels.extend(
            [label]
            * n
        )

        base_participants.extend(
            [participant]
            * n
        )

        base_phases.extend(
            [phase]
            * n
        )

        base_source_archive.extend(
            [path.name]
            * n
        )

        base_source_index.extend(
            list(
                range(
                    n
                )
            )
        )

        archive_sizes[
            path.name
        ] = n


    eeg = np.concatenate(
        base_eeg,
        axis=0,
    )

    ecg = np.concatenate(
        base_ecg,
        axis=0,
    )

    pupil = np.concatenate(
        base_pupil,
        axis=0,
    )

    pupil_valid = np.concatenate(
        base_pupil_valid,
        axis=0,
    )

    labels = np.asarray(
        base_labels,
        dtype=np.int64,
    )

    participants = np.asarray(
        base_participants,
        dtype="U3",
    )

    phases = np.asarray(
        base_phases,
        dtype=np.int16,
    )

    source_archives = np.asarray(
        base_source_archive,
        dtype="U32",
    )

    source_indices = np.asarray(
        base_source_index,
        dtype=np.int32,
    )


    if eeg.shape != (
        3315,
        8,
        512,
    ):
        raise RuntimeError(
            f"Internal EEG cache shape mismatch: {eeg.shape}"
        )

    if ecg.shape != (
        3315,
        1,
        512,
    ):
        raise RuntimeError(
            f"Internal ECG cache shape mismatch: {ecg.shape}"
        )

    if pupil.shape != (
        3315,
        1,
        120,
    ):
        raise RuntimeError(
            f"Internal pupil cache shape mismatch: {pupil.shape}"
        )

    if int(
        pupil_valid.sum()
    ) != 2306:
        raise RuntimeError(
            "Internal base pupil-valid count mismatch."
        )


    base_cache = (
        CACHE_INTERNAL
        / "window_04s.npz"
    )

    np.savez_compressed(
        base_cache,
        eeg=eeg,
        ecg=ecg,
        pupil=pupil,
        pupil_valid=pupil_valid,
        label=labels,
        participant=participants,
        phase=phases,
        source_archive=source_archives,
        source_window_index=source_indices,
        window_seconds=np.asarray(
            4,
            dtype=np.int16,
        ),
    )


    duration_summaries = {
        "4": {
            "candidate_windows":
                3315,

            "pupil_valid_windows":
                2306,

            "eeg_shape":
                list(
                    eeg.shape
                ),

            "ecg_shape":
                list(
                    ecg.shape
                ),

            "pupil_shape":
                list(
                    pupil.shape
                ),

            "cache":
                str(
                    base_cache.relative_to(
                        ROOT
                    )
                ),
        }
    }

    cache_rows = [
        {
            "dataset":
                "internal_xr",

            "duration_seconds":
                4,

            "unit":
                "all_base_windows",

            "recording_id":
                "ALL",

            "candidate_windows":
                3315,

            "accepted_pupil_windows":
                2306,

            "cache_path":
                str(
                    base_cache.relative_to(
                        ROOT
                    )
                ),

            "sha256":
                sha256_file(
                    base_cache
                ),
        }
    ]


    # Build longer candidate windows strictly within participant+phase
    # source archives from contiguous audited 4-s windows.
    for duration in [
        8,
        16,
        24,
        32,
    ]:

        constituent = (
            duration
            // 4
        )

        stride = (
            constituent
            // 2
        )

        long_eeg = []
        long_ecg = []
        long_pupil = []
        long_pupil_valid = []
        long_label = []
        long_participant = []
        long_phase = []
        long_archive = []
        long_start_index = []

        offset = 0

        for archive_path in archive_paths:

            n = archive_sizes[
                archive_path.name
            ]

            archive_slice = slice(
                offset,
                offset + n,
            )

            archive_eeg = eeg[
                archive_slice
            ]

            archive_ecg = ecg[
                archive_slice
            ]

            archive_pupil = pupil[
                archive_slice
            ]

            archive_pupil_valid = pupil_valid[
                archive_slice
            ]

            archive_label = labels[
                archive_slice
            ]

            archive_participant = participants[
                archive_slice
            ]

            archive_phase = phases[
                archive_slice
            ]

            for start in range(
                0,
                n - constituent + 1,
                stride,
            ):

                stop = (
                    start
                    + constituent
                )

                long_eeg.append(
                    np.concatenate(
                        archive_eeg[
                            start:stop
                        ],
                        axis=-1,
                    )
                )

                long_ecg.append(
                    np.concatenate(
                        archive_ecg[
                            start:stop
                        ],
                        axis=-1,
                    )
                )

                long_pupil.append(
                    np.concatenate(
                        archive_pupil[
                            start:stop
                        ],
                        axis=-1,
                    )
                )

                valid = bool(
                    archive_pupil_valid[
                        start:stop
                    ].all()
                )

                long_pupil_valid.append(
                    valid
                )

                unique_labels = np.unique(
                    archive_label[
                        start:stop
                    ]
                )

                if len(
                    unique_labels
                ) != 1:
                    raise RuntimeError(
                        "Internal long window crossed label boundary."
                    )

                long_label.append(
                    int(
                        unique_labels[
                            0
                        ]
                    )
                )

                long_participant.append(
                    str(
                        archive_participant[
                            start
                        ]
                    )
                )

                long_phase.append(
                    int(
                        archive_phase[
                            start
                        ]
                    )
                )

                long_archive.append(
                    archive_path.name
                )

                long_start_index.append(
                    start
                )

            offset += n


        long_eeg = np.stack(
            long_eeg
        ).astype(
            np.float32,
            copy=False,
        )

        long_ecg = np.stack(
            long_ecg
        ).astype(
            np.float32,
            copy=False,
        )

        long_pupil = np.stack(
            long_pupil
        ).astype(
            np.float32,
            copy=False,
        )

        long_pupil_valid = np.asarray(
            long_pupil_valid,
            dtype=bool,
        )

        long_label = np.asarray(
            long_label,
            dtype=np.int64,
        )

        long_participant = np.asarray(
            long_participant,
            dtype="U3",
        )

        long_phase = np.asarray(
            long_phase,
            dtype=np.int16,
        )

        long_archive = np.asarray(
            long_archive,
            dtype="U32",
        )

        long_start_index = np.asarray(
            long_start_index,
            dtype=np.int32,
        )


        expected = {
            8:
                (
                    3276,
                    1962,
                ),

            16:
                (
                    1599,
                    767,
                ),

            24:
                (
                    1066,
                    433,
                ),

            32:
                (
                    767,
                    267,
                ),
        }[
            duration
        ]

        if long_eeg.shape[
            0
        ] != expected[
            0
        ]:
            raise RuntimeError(
                f"{duration}s internal candidate count mismatch: "
                f"{long_eeg.shape[0]}"
            )

        if int(
            long_pupil_valid.sum()
        ) != expected[
            1
        ]:
            raise RuntimeError(
                f"{duration}s internal pupil-valid count mismatch: "
                f"{int(long_pupil_valid.sum())}"
            )


        cache_path = (
            CACHE_INTERNAL
            / f"window_{duration:02d}s.npz"
        )

        np.savez_compressed(
            cache_path,
            eeg=long_eeg,
            ecg=long_ecg,
            pupil=long_pupil,
            pupil_valid=long_pupil_valid,
            label=long_label,
            participant=long_participant,
            phase=long_phase,
            source_archive=long_archive,
            source_start_4s_index=long_start_index,
            window_seconds=np.asarray(
                duration,
                dtype=np.int16,
            ),
            stride_seconds=np.asarray(
                duration // 2,
                dtype=np.int16,
            ),
        )


        duration_summaries[
            str(
                duration
            )
        ] = {
            "candidate_windows":
                int(
                    len(
                        long_label
                    )
                ),

            "pupil_valid_windows":
                int(
                    long_pupil_valid.sum()
                ),

            "eeg_shape":
                list(
                    long_eeg.shape
                ),

            "ecg_shape":
                list(
                    long_ecg.shape
                ),

            "pupil_shape":
                list(
                    long_pupil.shape
                ),

            "cache":
                str(
                    cache_path.relative_to(
                        ROOT
                    )
                ),
        }

        cache_rows.append(
            {
                "dataset":
                    "internal_xr",

                "duration_seconds":
                    duration,

                "unit":
                    "all_candidate_windows",

                "recording_id":
                    "ALL",

                "candidate_windows":
                    int(
                        len(
                            long_label
                        )
                    ),

                "accepted_pupil_windows":
                    int(
                        long_pupil_valid.sum()
                    ),

                "cache_path":
                    str(
                        cache_path.relative_to(
                            ROOT
                        )
                    ),

                "sha256":
                    sha256_file(
                        cache_path
                    ),
            }
        )

        print(
            f"{duration:>2}s internal cache: "
            f"{len(long_label):,} candidates, "
            f"{int(long_pupil_valid.sum()):,} pupil-valid"
        )


    return (
        duration_summaries,
        cache_rows,
    )


def robust_bool(
    series: pd.Series,
) -> pd.Series:

    if series.dtype == bool:
        return series

    return (
        series
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(
            {
                "true",
                "1",
                "yes",
                "y",
            }
        )
    )


def resolve_bbbd_metadata_columns(
    frame: pd.DataFrame,
):

    required = [
        "recording_id",
        "dataset",
        "participant",
        "subject",
        "session",
        "task",
        "label",
        "start_seconds",
        "end_seconds",
    ]

    missing = [
        column
        for column in required
        if column not in frame.columns
    ]

    if missing:
        raise RuntimeError(
            "BBBD feature table missing required metadata columns: "
            f"{missing}"
        )


def build_bbbd_cache(bbbd_archives: dict[str, Path]):
    print(
        "\n===== BBBD COMPLETE 392-RECORDING CACHE ====="
    )

    CACHE_BBBD.mkdir(
        parents=True,
        exist_ok=False,
    )

    config = load_bbbd_config(
        "configs/bbbd.yaml"
    )

    fs = float(
        config[
            "signals"
        ][
            "sampling_frequency_hz"
        ]
    )

    if not math.isclose(
        fs,
        128.0,
        abs_tol=1.0e-9,
    ):
        raise RuntimeError(
            "Expected BBBD model sampling rate 128 Hz."
        )


    accepted = pd.read_csv(
        BBBD_FEATURE_TABLE,
        low_memory=False,
    )

    resolve_bbbd_metadata_columns(
        accepted
    )

    audit = pd.read_csv(
        BBBD_RECORDING_AUDIT,
        low_memory=False,
    )

    if len(
        audit
    ) != 392:
        raise RuntimeError(
            f"Expected 392 BBBD recordings, found {len(audit)}."
        )

    if len(
        accepted
    ) != 51831:
        raise RuntimeError(
            f"Expected 51,831 accepted BBBD windows, "
            f"found {len(accepted)}."
        )


    manifests = []

    for path in [
        BBBD_EXP2_MANIFEST,
        BBBD_EXP3_MANIFEST,
    ]:

        frame = pd.read_csv(
            path,
            low_memory=False,
        )

        manifests.append(
            frame
        )

    records = pd.concat(
        manifests,
        ignore_index=True,
    )


    cohort = pd.read_csv(
        BBBD_COHORT,
        low_memory=False,
    )

    direct = cohort.loc[
        robust_bool(
            cohort[
                "direct_complete_case"
            ]
        )
    ].copy()

    if len(
        direct
    ) != 36:
        raise RuntimeError(
            "Expected 36 direct-comparison BBBD participants."
        )

    direct_participants = set(
        direct[
            "participant"
        ].astype(str)
    )


    audit_participants = set(
        audit[
            "participant"
        ].astype(str)
    )

    if audit_participants != direct_participants:
        raise RuntimeError(
            "BBBD recording-audit participant universe "
            "does not equal locked direct cohort."
        )


    expected_recording_counts = {
        str(
            row.recording_id
        ):
            (
                int(
                    row.candidate_windows
                ),
                int(
                    row.accepted_windows
                ),
            )
        for row
        in audit.itertuples(
            index=False
        )
    }


    feature_group_counts = (
        accepted[
            "recording_id"
        ]
        .astype(str)
        .value_counts()
        .to_dict()
    )

    for recording_id, (
        candidate_count,
        accepted_count,
    ) in expected_recording_counts.items():

        actual = int(
            feature_group_counts.get(
                recording_id,
                0,
            )
        )

        if actual != accepted_count:
            raise RuntimeError(
                f"{recording_id}: accepted feature-row count "
                f"{actual} != locked audit {accepted_count}."
            )


    progress = Progress(
        total=len(
            audit
        ),
        description=
            "BBBD raw cache",
    )

    cache_rows = []

    total_cached = 0
    label_counts = {}

    try:

        for audit_row in audit.itertuples(
            index=False
        ):

            recording_id = str(
                audit_row.recording_id
            )

            dataset = str(
                audit_row.dataset
            )

            subject = str(
                audit_row.subject
            )

            session = str(
                audit_row.session
            )

            task = str(
                audit_row.task
            )

            label = int(
                audit_row.label
            )

            participant = str(
                audit_row.participant
            )


            feature_rows = accepted.loc[
                accepted[
                    "recording_id"
                ].astype(str)
                == recording_id
            ].copy()

            feature_rows = feature_rows.sort_values(
                "start_seconds"
            ).reset_index(
                drop=True
            )


            manifest_rows = records.loc[
                (
                    records[
                        "subject"
                    ].astype(str)
                    == subject
                )
                & (
                    records[
                        "session"
                    ].astype(str)
                    == session
                )
                & (
                    records[
                        "task"
                    ].astype(str)
                    == task
                )
                & (
                    records[
                        "dataset"
                    ].astype(str)
                    == (
                        "bbbd_experiment2"
                        if dataset
                        == "experiment2"
                        else "bbbd_experiment3"
                    )
                )
            ]

            if len(
                manifest_rows
            ) != 1:
                raise RuntimeError(
                    f"{recording_id}: expected one manifest row, "
                    f"found {len(manifest_rows)}."
                )

            manifest_row = manifest_rows.iloc[
                0
            ]


            archive_path = bbbd_archives[
                dataset
            ]

            with zipfile.ZipFile(
                archive_path,
                "r",
            ) as archive:

                eeg_member = derivative_eeg_member(
                    str(
                        manifest_row[
                            "eeg_path"
                        ]
                    )
                )

                with tempfile.TemporaryDirectory() as temp_dir:

                    temp_bdf = (
                        Path(
                            temp_dir
                        )
                        / "recording.bdf"
                    )

                    temp_bdf.write_bytes(
                        archive.read(
                            eeg_member
                        )
                    )

                    raw = mne.io.read_raw_bdf(
                        temp_bdf,
                        preload=True,
                        verbose="ERROR",
                    )

                    if any(
                        channel not in raw.ch_names
                        for channel
                        in EEG_CHANNELS
                    ):
                        raise RuntimeError(
                            f"{recording_id}: required EEG channel missing."
                        )

                    eeg_full = raw.get_data(
                        picks=
                            EEG_CHANNELS
                    )

                    del raw


                if config[
                    "signals"
                ][
                    "eeg"
                ][
                    "convert_volts_to_microvolts"
                ]:

                    eeg_full = (
                        eeg_full
                        * 1.0e6
                    )


                ecg_raw = read_headerless_tsv_gz(
                    archive,
                    str(
                        manifest_row[
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
                        config[
                            "signals"
                        ][
                            "ecg"
                        ][
                            "bandpass_hz"
                        ],
                    filter_order=
                        int(
                            config[
                                "signals"
                            ][
                                "ecg"
                            ][
                                "filter_order"
                            ]
                        ),
                    minimum_finite_fraction=
                        float(
                            config[
                                "signals"
                            ][
                                "ecg"
                            ][
                                "minimum_finite_fraction"
                            ]
                        ),
                )


                pupil_raw = read_headerless_tsv_gz(
                    archive,
                    str(
                        manifest_row[
                            "pupil_path"
                        ]
                    ),
                    "pupil_size",
                )

                maximum_gap_samples = int(
                    round(
                        float(
                            config[
                                "signals"
                            ][
                                "pupil"
                            ][
                                "maximum_interpolation_gap_seconds"
                            ]
                        )
                        * fs
                    )
                )

                pupil_full, original_valid = preprocess_pupil(
                    pupil_raw,
                    minimum_valid_value_exclusive=
                        float(
                            config[
                                "signals"
                            ][
                                "pupil"
                            ][
                                "minimum_valid_value_exclusive"
                            ]
                        ),
                    maximum_gap_samples=
                        maximum_gap_samples,
                )


            eeg_windows = []
            ecg_windows = []
            pupil_windows = []
            starts = []
            ends = []
            valid_fractions = []

            for row in feature_rows.itertuples(
                index=False
            ):

                start_seconds = float(
                    row.start_seconds
                )

                end_seconds = float(
                    row.end_seconds
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

                if (
                    stop
                    - start
                ) != 512:
                    raise RuntimeError(
                        f"{recording_id}: non-512-sample accepted window."
                    )


                eeg_window = np.asarray(
                    eeg_full[
                        :,
                        start:stop
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

                valid_fraction = float(
                    np.mean(
                        original_valid[
                            start:stop
                        ]
                    )
                )


                if eeg_window.shape != (
                    8,
                    512,
                ):
                    raise RuntimeError(
                        f"{recording_id}: EEG window shape mismatch."
                    )

                if ecg_window.shape != (
                    512,
                ):
                    raise RuntimeError(
                        f"{recording_id}: ECG window shape mismatch."
                    )

                if pupil_window.shape != (
                    512,
                ):
                    raise RuntimeError(
                        f"{recording_id}: pupil window shape mismatch."
                    )

                if not np.isfinite(
                    eeg_window
                ).all():
                    raise RuntimeError(
                        f"{recording_id}: nonfinite EEG accepted window."
                    )

                if not np.isfinite(
                    ecg_window
                ).all():
                    raise RuntimeError(
                        f"{recording_id}: nonfinite ECG accepted window."
                    )

                if not np.isfinite(
                    pupil_window
                ).all():
                    raise RuntimeError(
                        f"{recording_id}: nonfinite pupil accepted window."
                    )

                if valid_fraction < 0.8:
                    raise RuntimeError(
                        f"{recording_id}: accepted pupil valid fraction "
                        f"{valid_fraction:.6f} < 0.8."
                    )


                eeg_windows.append(
                    eeg_window
                )

                ecg_windows.append(
                    ecg_window[
                        None,
                        :
                    ]
                )

                pupil_windows.append(
                    pupil_window[
                        None,
                        :
                    ]
                )

                starts.append(
                    start_seconds
                )

                ends.append(
                    end_seconds
                )

                valid_fractions.append(
                    valid_fraction
                )


            eeg_windows = np.stack(
                eeg_windows
            ).astype(
                np.float32,
                copy=False,
            )

            ecg_windows = np.stack(
                ecg_windows
            ).astype(
                np.float32,
                copy=False,
            )

            pupil_windows = np.stack(
                pupil_windows
            ).astype(
                np.float32,
                copy=False,
            )


            expected_accepted = int(
                audit_row.accepted_windows
            )

            if eeg_windows.shape[
                0
            ] != expected_accepted:
                raise RuntimeError(
                    f"{recording_id}: raw cache count mismatch."
                )


            safe_id = recording_id.replace(
                "::",
                "__",
            )

            dataset_root = (
                CACHE_BBBD
                / dataset
            )

            dataset_root.mkdir(
                parents=True,
                exist_ok=True,
            )

            cache_path = (
                dataset_root
                / f"{safe_id}.npz"
            )


            np.savez_compressed(
                cache_path,
                eeg=
                    eeg_windows,

                ecg=
                    ecg_windows,

                pupil=
                    pupil_windows,

                label=
                    np.full(
                        expected_accepted,
                        label,
                        dtype=np.int64,
                    ),

                participant=
                    np.asarray(
                        [participant]
                        * expected_accepted,
                        dtype="U32",
                    ),

                recording_id=
                    np.asarray(
                        [recording_id]
                        * expected_accepted,
                        dtype="U64",
                    ),

                start_seconds=
                    np.asarray(
                        starts,
                        dtype=np.float64,
                    ),

                end_seconds=
                    np.asarray(
                        ends,
                        dtype=np.float64,
                    ),

                pupil_valid_fraction=
                    np.asarray(
                        valid_fractions,
                        dtype=np.float32,
                    ),

                window_seconds=
                    np.asarray(
                        4,
                        dtype=np.int16,
                    ),

                sampling_frequency_hz=
                    np.asarray(
                        128.0,
                        dtype=np.float32,
                    ),
            )


            total_cached += expected_accepted

            label_counts[
                label
            ] = (
                label_counts.get(
                    label,
                    0,
                )
                + expected_accepted
            )


            cache_rows.append(
                {
                    "dataset":
                        dataset,

                    "duration_seconds":
                        4,

                    "unit":
                        "recording",

                    "recording_id":
                        recording_id,

                    "participant":
                        participant,

                    "label":
                        label,

                    "candidate_windows":
                        int(
                            audit_row.candidate_windows
                        ),

                    "accepted_pupil_windows":
                        expected_accepted,

                    "cache_path":
                        str(
                            cache_path.relative_to(
                                ROOT
                            )
                        ),

                    "sha256":
                        sha256_file(
                            cache_path
                        ),
                }
            )


            progress.update(
                context=
                    (
                        f"{dataset} | "
                        f"{recording_id} | "
                        f"cached={total_cached:,}"
                    )
            )

    finally:
        progress.close()


    if total_cached != 51831:
        raise RuntimeError(
            f"Expected 51,831 BBBD cached windows, "
            f"found {total_cached}."
        )


    authoritative_label_counts = (
        accepted[
            "label"
        ]
        .astype(int)
        .value_counts()
        .sort_index()
        .to_dict()
    )

    if label_counts != authoritative_label_counts:
        raise RuntimeError(
            "BBBD raw-cache label counts do not match "
            "authoritative accepted feature table."
        )


    authoritative_dataset_counts = (
        accepted[
            "dataset"
        ]
        .astype(str)
        .value_counts()
        .sort_index()
        .to_dict()
    )

    cached_dataset_counts = {}

    for row in cache_rows:

        cached_dataset_counts[
            row[
                "dataset"
            ]
        ] = (
            cached_dataset_counts.get(
                row[
                    "dataset"
                ],
                0,
            )
            + int(
                row[
                    "accepted_pupil_windows"
                ]
            )
        )

    if cached_dataset_counts != authoritative_dataset_counts:
        raise RuntimeError(
            "BBBD dataset-window counts do not match authoritative table."
        )


    return (
        {
            "recordings":
                392,

            "participants":
                36,

            "candidate_windows":
                int(
                    audit[
                        "candidate_windows"
                    ].sum()
                ),

            "accepted_windows":
                total_cached,

            "rejected_pupil_windows":
                int(
                    audit[
                        "rejected_pupil_windows"
                    ].sum()
                ),

            "label_counts":
                {
                    str(key):
                        int(value)
                    for key, value
                    in sorted(
                        label_counts.items()
                    )
                },

            "dataset_window_counts":
                {
                    str(key):
                        int(value)
                    for key, value
                    in sorted(
                        cached_dataset_counts.items()
                    )
                },

            "sampling_frequency_hz":
                128.0,

            "eeg_channels":
                EEG_CHANNELS,

            "eeg_window_shape":
                [
                    8,
                    512,
                ],

            "ecg_window_shape":
                [
                    1,
                    512,
                ],

            "pupil_window_shape":
                [
                    1,
                    512,
                ],

            "pupil_semantics":
                "pupil_size only",
        },
        cache_rows,
    )


def verify_cache_files(
    rows: list[dict],
):

    print(
        "\n===== VERIFY CACHE FILE CHECKSUMS ====="
    )

    for row in rows:

        path = (
            ROOT
            / row[
                "cache_path"
            ]
        )

        if not path.is_file():
            raise FileNotFoundError(
                path
            )

        actual = sha256_file(
            path
        )

        if actual != row[
            "sha256"
        ]:
            raise RuntimeError(
                f"Cache checksum mismatch: {path}"
            )

    print(
        "Cache files verified:",
        len(
            rows
        ),
    )


def main():

    project_config = load_revision_config(
        check_input_paths=False
    )

    bbbd_archives = {
        "experiment2": Path(
            project_config["paths"]["bbbd_experiment2_archive"]
        ),
        "experiment3": Path(
            project_config["paths"]["bbbd_experiment3_archive"]
        ),
    }

    for dataset, archive_path in bbbd_archives.items():
        if not archive_path.is_file():
            raise FileNotFoundError(
                f"Configured BBBD archive does not exist: "
                f"{dataset} = {archive_path}"
            )

    ensure_clean_target()

    protocol = load_protocol()

    start_time = time.time()


    internal_summary, internal_rows = (
        build_internal_cache()
    )

    bbbd_summary, bbbd_rows = (
        build_bbbd_cache(
            bbbd_archives
        )
    )


    all_rows = (
        internal_rows
        + bbbd_rows
    )

    verify_cache_files(
        all_rows
    )


    cache_manifest = pd.DataFrame(
        all_rows
    )

    CACHE_MANIFEST.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    cache_manifest.to_csv(
        CACHE_MANIFEST,
        index=False,
    )


    if int(
        bbbd_summary[
            "candidate_windows"
        ]
    ) != 53512:
        raise RuntimeError(
            "BBBD total candidate-window parity failed."
        )

    if int(
        bbbd_summary[
            "accepted_windows"
        ]
    ) != 51831:
        raise RuntimeError(
            "BBBD total accepted-window parity failed."
        )

    if int(
        bbbd_summary[
            "rejected_pupil_windows"
        ]
    ) != 1681:
        raise RuntimeError(
            "BBBD total pupil-rejection parity failed."
        )


    elapsed = (
        time.time()
        - start_time
    )


    report = {
        "identity":
            "deep_learning_raw_cache_audit_v1",

        "protocol_revision":
            "1.2",

        "cache_root":
            str(
                CACHE_ROOT.relative_to(
                    ROOT
                )
            ),

        "real_research_data_loaded":
            True,

        "normalization_statistics_fitted":
            False,

        "optimizer_created":
            False,

        "loss_computed":
            False,

        "backward_pass_performed":
            False,

        "neural_network_fitting_performed":
            False,

        "predictions_generated":
            False,

        "performance_metrics_computed":
            False,

        "performance_observed":
            False,

        "internal":
            internal_summary,

        "bbbd":
            bbbd_summary,

        "cache_files":
            len(
                all_rows
            ),

        "cache_manifest":
            str(
                CACHE_MANIFEST.relative_to(
                    ROOT
                )
            ),

        "cache_manifest_sha256":
            sha256_file(
                CACHE_MANIFEST
            ),

        "cache_file_checksums_verified":
            True,

        "full_universe_parity_passed":
            True,

        "elapsed_seconds":
            elapsed,

        "execution_progress_display":
            True,

        "next_gate":
            (
                "training-runner implementation, training-only "
                "normalization tests, adaptive split verification, "
                "and synthetic/minimal no-performance execution smoke"
            ),

        "real_training_allowed_after_cache_only":
            False,
    }


    AUDIT_JSON.write_text(
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


    md = [
        "# Deep-Learning Raw-Input Cache Audit v1",
        "",
        "## Status",
        "",
        "**PASSED  complete raw-input universe cached.**",
        "",
        "No neural-network fitting or performance evaluation was performed.",
        "",
        "## Internal XR",
        "",
        "- Participants: 13.",
        "- Base 4-s windows: 3,315.",
        "- Pupil-valid base windows: 2,306.",
        "- Pupil input: internal `r` only.",
        "",
        "| Duration | Candidate windows | Pupil-valid windows |",
        "|---:|---:|---:|",
    ]

    for duration in [
        "4",
        "8",
        "16",
        "24",
        "32",
    ]:

        item = internal_summary[
            duration
        ]

        md.append(
            f"| {duration} s | "
            f"{item['candidate_windows']:,} | "
            f"{item['pupil_valid_windows']:,} |"
        )


    md.extend(
        [
            "",
            "## BBBD",
            "",
            f"- Participants: {bbbd_summary['participants']}.",
            f"- Recordings: {bbbd_summary['recordings']}.",
            f"- Candidate windows: {bbbd_summary['candidate_windows']:,}.",
            f"- Accepted aligned windows: {bbbd_summary['accepted_windows']:,}.",
            f"- Pupil-rejected windows: {bbbd_summary['rejected_pupil_windows']:,}.",
            "- EEG: eight locked channels at 128 Hz.",
            "- ECG: one channel at 128 Hz.",
            "- Pupil: scalar `pupil_size` at 128 Hz.",
            "",
            "## Cache integrity",
            "",
            f"- Cache files: {len(all_rows)}.",
            "- Every cache file SHA-256 verified.",
            "- Internal candidate counts match the locked preprocessing audit.",
            "- BBBD recording-wise accepted counts match the locked recording audit.",
            "- BBBD total candidate/accepted/rejected counts match exactly.",
            "- BBBD label and dataset window distributions match the authoritative accepted feature table.",
            "",
            "## Performance firewall",
            "",
            "- Training normalization fitted: no.",
            "- Optimizer created: no.",
            "- Loss computed: no.",
            "- Backward pass: no.",
            "- Neural-network fitting: no.",
            "- Predictions generated: no.",
            "- Performance metrics computed: no.",
            "",
            f"Cache-construction elapsed time: {elapsed / 60:.2f} minutes.",
        ]
    )


    AUDIT_MD.write_text(
        "\n".join(
            md
        ).rstrip()
        + "\n",
        encoding="utf-8",
    )


    print(
        "\n===== COMPLETE CACHE VERDICT ====="
    )

    print(
        "Internal 4-s windows: 3,315 / 3,315"
    )

    print(
        "Internal 8-s windows: 3,276 / 3,276"
    )

    print(
        "Internal 16-s windows: 1,599 / 1,599"
    )

    print(
        "Internal 24-s windows: 1,066 / 1,066"
    )

    print(
        "Internal 32-s windows: 767 / 767"
    )

    print(
        "BBBD recordings: 392 / 392"
    )

    print(
        "BBBD candidate windows: 53,512 / 53,512"
    )

    print(
        "BBBD accepted windows: 51,831 / 51,831"
    )

    print(
        "BBBD rejected pupil windows: 1,681 / 1,681"
    )

    print(
        "Cache checksum verification: PASSED"
    )

    print(
        "Full-universe parity: PASSED"
    )

    print(
        "Neural-network fitting performed: False"
    )

    print(
        "Performance observed: False"
    )

    print(
        "Real training allowed now: False"
    )

    print(
        "Elapsed minutes:",
        round(
            elapsed / 60,
            2,
        ),
    )

    print(
        "Created:",
        AUDIT_JSON.relative_to(
            ROOT
        ),
    )

    print(
        "Created:",
        AUDIT_MD.relative_to(
            ROOT
        ),
    )

    print(
        "Created:",
        CACHE_MANIFEST.relative_to(
            ROOT
        ),
    )


if __name__ == "__main__":
    main()
