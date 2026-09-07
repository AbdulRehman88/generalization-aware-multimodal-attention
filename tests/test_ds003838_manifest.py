"""Tests for the ds003838 read-only manifest scanner."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import src.data_loaders.ds003838_manifest as manifest_module

from src.data_loaders.ds003838_manifest import (
    _clock_segments,
    _decode_code,
    _timestamp_coverage_segments,
    scan_ds003838,
)


REQUIRED_CHANNELS = [
    "AF7",
    "Fp1",
    "Fp2",
    "AF8",
    "O1",
    "POz",
    "O2",
]


class DS003838ManifestTests(unittest.TestCase):
    def test_event_code_decoding(self) -> None:
        memory = _decode_code(6005091)
        control = _decode_code(501013)

        self.assertEqual(memory["condition"], "memory")
        self.assertEqual(memory["position"], 5)
        self.assertEqual(memory["length"], 9)
        self.assertEqual(memory["correct"], 1)

        self.assertEqual(control["condition"], "control")
        self.assertEqual(control["position"], 10)
        self.assertEqual(control["length"], 13)


    def test_trial_final_offset_includes_event_duration(self) -> None:
        events = pd.DataFrame(
            [
                {
                    "code": 6001050,
                    "time": 10.0,
                    "duration": 1.0,
                    "condition": "memory",
                    "position": 1,
                    "length": 5,
                },
                {
                    "code": 6002050,
                    "time": 12.0,
                    "duration": 1.0,
                    "condition": "memory",
                    "position": 2,
                    "length": 5,
                },
                {
                    "code": 6003050,
                    "time": 14.0,
                    "duration": 1.0,
                    "condition": "memory",
                    "position": 3,
                    "length": 5,
                },
                {
                    "code": 6004050,
                    "time": 16.0,
                    "duration": 1.0,
                    "condition": "memory",
                    "position": 4,
                    "length": 5,
                },
                {
                    "code": 6005050,
                    "time": 18.0,
                    "duration": 1.0,
                    "condition": "memory",
                    "position": 5,
                    "length": 5,
                },
            ]
        )

        trials = manifest_module._build_trials(events)

        self.assertEqual(len(trials), 1)
        self.assertAlmostEqual(
            trials[0]["final_time"],
            18.0,
        )
        self.assertAlmostEqual(
            trials[0]["final_duration"],
            1.0,
        )
        self.assertAlmostEqual(
            trials[0]["final_offset"],
            19.0,
        )

    def test_clock_jump_creates_two_segments(self) -> None:
        eeg_times = np.arange(10, dtype=float)

        pupil_times = eeg_times + 1000.0
        pupil_times[5:] += 4.0

        segments, jumps = _clock_segments(
            eeg_times,
            pupil_times,
            jump_threshold_seconds=0.25,
        )

        self.assertEqual(len(segments), 2)
        self.assertEqual(
            int((np.abs(jumps) > 0.25).sum()),
            1,
        )

        self.assertLess(
            segments[0]["maximum_absolute_residual"],
            1e-9,
        )
        self.assertLess(
            segments[1]["maximum_absolute_residual"],
            1e-9,
        )


    def test_internal_pupil_gap_creates_two_segments(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "pupil.tsv"

            path.write_text(
                "pupil_timestamp\tconfidence\n"
                "1000.0\t1.0\n"
                "1000.1\t1.0\n"
                "1000.2\t1.0\n"
                "1002.0\t1.0\n"
                "1002.1\t1.0\n",
                encoding="utf-8",
            )

            segments, maximum_gap = (
                _timestamp_coverage_segments(
                    path,
                    maximum_gap_seconds=0.5,
                )
            )

            self.assertEqual(
                segments,
                [
                    (1000.0, 1000.2),
                    (1002.0, 1002.1),
                ],
            )

            self.assertAlmostEqual(
                maximum_gap,
                1.8,
                places=6,
            )

    def test_complete_subject_is_strict_primary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            subject = "sub-001"
            subject_root = root / subject

            for directory in (
                subject_root / "eeg",
                subject_root / "ecg",
                subject_root / "pupil",
                subject_root / "beh",
            ):
                directory.mkdir(parents=True)

            participants_text = (
                "participant_id\tEEG_excluded\tECG_excluded\t"
                "pupil_excluded\tbehavior_excluded\n"
                f"{subject}\tno\tno\tno\tno\n"
            )

            (root / "participants.tsv").write_text(
                participants_text,
                encoding="utf-8",
            )

            for path in (
                subject_root / "eeg" /
                f"{subject}_task-memory_eeg.set",

                subject_root / "ecg" /
                f"{subject}_task-memory_ecg.set",

                subject_root / "beh" /
                f"{subject}_task-memory_beh.tsv",
            ):
                path.write_bytes(b"synthetic")

            channel_text = (
                "name\ttype\tunits\n"
                + "\n".join(
                    f"{channel}\tEEG\tuV"
                    for channel in REQUIRED_CHANNELS
                )
                + "\n"
            )

            (
                subject_root / "eeg" /
                f"{subject}_task-memory_channels.tsv"
            ).write_text(
                channel_text,
                encoding="utf-8",
            )

            eeg_rows = [
                (
                    "onset\tduration\tsample\ttrial_type\t"
                    "response_time\tstim_file\tvalue"
                )
            ]

            pupil_event_rows = [
                "index\ttimestamp\tlabel"
            ]

            onset = 10.0
            event_index = 0

            for length in (5, 9, 13):
                for position in range(1, length + 1):
                    code = int(
                        f"6{position:03d}{length:02d}1"
                    )

                    trial_type = (
                        f"memory {position:02d}/{length:02d}"
                    )

                    eeg_rows.append(
                        f"{onset}\t1\t{int(onset * 1000)}\t"
                        f"{trial_type}\tn/a\tn/a\t{code}"
                    )

                    pupil_event_rows.append(
                        f"{event_index}\t{1000.0 + onset}\t{code}"
                    )

                    onset += 2.0
                    event_index += 1

                onset += 6.0

            (
                subject_root / "eeg" /
                f"{subject}_task-memory_events.tsv"
            ).write_text(
                "\n".join(eeg_rows) + "\n",
                encoding="utf-8",
            )

            (
                subject_root / "pupil" /
                f"{subject}_task-memory_events.tsv"
            ).write_text(
                "\n".join(pupil_event_rows) + "\n",
                encoding="utf-8",
            )

            pupil_sample_lines = [
                "pupil_timestamp\tconfidence"
            ]

            pupil_sample_lines.extend(
                f"{timestamp:.1f}\t1.0"
                for timestamp in np.arange(
                    1000.0,
                    1200.1,
                    0.1,
                )
            )

            (
                subject_root / "pupil" /
                f"{subject}_task-memory_pupil.tsv"
            ).write_text(
                "\n".join(pupil_sample_lines) + "\n",
                encoding="utf-8",
            )

            records, summary = scan_ds003838(
                root,
                required_channels=REQUIRED_CHANNELS,
                expected_trials_per_class=1,
            )

            self.assertEqual(len(records), 1)
            self.assertTrue(
                records[0]["analysis_eligible"]
            )
            self.assertTrue(
                records[0]["complete_balanced_108"]
            )
            self.assertTrue(
                records[0]["strict_primary_cohort"]
            )
            self.assertEqual(
                records[0]["usable_memory_total"],
                3,
            )
            self.assertEqual(
                summary["strict_primary_subject_count"],
                1,
            )
            self.assertEqual(
                summary["strict_primary_window_count"],
                3,
            )


if __name__ == "__main__":
    unittest.main()
