"""Tests for the BBBD read-only manifest scanner."""

from __future__ import annotations

import tempfile
import unittest
import zipfile
from pathlib import Path

from src.data_loaders.bbbd_manifest import (
    _expected_task_ids,
    _match_raw_path,
    scan_bbbd_archive,
)


REQUIRED_CHANNELS = [
    "AF7",
    "Fp1",
    "Fpz",
    "Fp2",
    "AF8",
    "O1",
    "POz",
    "O2",
]


class BBBDManifestTests(unittest.TestCase):
    def test_path_parser_ignores_derivatives(self) -> None:
        raw_path = (
            "sub-01/ses-01/eeg/"
            "sub-01_ses-01_task-stim01_eeg.bdf"
        )
        derivative_path = (
            "derivatives/sub-01/ses-01/eeg/"
            "sub-01_ses-01_task-stim01_desc-eeg.bdf"
        )

        self.assertEqual(
            _match_raw_path(raw_path),
            ("sub-01", "ses-01", "stim01", "eeg"),
        )
        self.assertIsNone(_match_raw_path(derivative_path))

    def test_expected_task_ids(self) -> None:
        self.assertEqual(
            _expected_task_ids(3),
            ["stim01", "stim02", "stim03"],
        )

    def test_complete_primary_subject_is_identified(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            archive_path = Path(temporary_directory) / "synthetic.zip"

            channel_text = (
                "name\ttype\tunits\n"
                + "\n".join(
                    f"{channel}\tEEG\tuV"
                    for channel in REQUIRED_CHANNELS
                )
                + "\n"
            )

            with zipfile.ZipFile(
                archive_path,
                "w",
                allowZip64=True,
            ) as archive:
                for session in ("ses-01", "ses-02"):
                    for task in ("stim01", "stim02"):
                        stem = f"sub-01_{session}_task-{task}"

                        archive.writestr(
                            f"sub-01/{session}/eeg/{stem}_eeg.bdf",
                            b"synthetic-bdf",
                        )
                        archive.writestr(
                            f"sub-01/{session}/eeg/"
                            f"{stem}_channels.tsv",
                            channel_text,
                        )
                        archive.writestr(
                            f"sub-01/{session}/beh/"
                            f"{stem}_recording-ecg_physio.tsv.gz",
                            b"synthetic-ecg",
                        )
                        archive.writestr(
                            f"sub-01/{session}/eyetrack/"
                            f"{stem}_pupil_eyetrack.tsv.gz",
                            b"synthetic-pupil",
                        )

            records, summary = scan_bbbd_archive(
                archive_path,
                dataset_name="synthetic",
                expected_tasks=2,
                required_channels=REQUIRED_CHANNELS,
                label_map={"ses-01": 1, "ses-02": 0},
            )

            self.assertEqual(len(records), 4)
            self.assertEqual(
                summary["complete_multimodal_records"],
                4,
            )
            self.assertEqual(
                summary["analysis_eligible_records"],
                4,
            )
            self.assertEqual(
                summary["primary_complete_subject_count"],
                1,
            )
            self.assertEqual(
                summary["primary_complete_record_count"],
                4,
            )
            self.assertTrue(
                all(record["primary_cohort"] for record in records)
            )


if __name__ == "__main__":
    unittest.main()