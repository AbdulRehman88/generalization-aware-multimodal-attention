"""Tests for the revision configuration resolver."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from src.core.config import load_revision_config


class RevisionConfigTests(unittest.TestCase):
    def test_local_configuration_overrides_and_resolves_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            local_path = directory / "config.local.yaml"

            local_configuration = {
                "paths": {
                    "internal_xr_raw": "temporary/internal/raw",
                    "internal_xr_legacy_processed": "temporary/internal/processed",
                    "bbbd_experiment2_archive": "temporary/experiment2.zip",
                    "bbbd_experiment3_archive": "temporary/experiment3.zip",
                    "ds003838_root": "temporary/ds003838",
                    "outputs": "temporary/outputs",
                    "artifacts": "temporary/artifacts",
                }
            }

            local_path.write_text(
                yaml.safe_dump(local_configuration),
                encoding="utf-8",
            )

            config = load_revision_config(
                local_path=local_path,
                check_input_paths=False,
            )

            self.assertEqual(
                config["evaluation"]["internal"]["primary"]["protocol"],
                "strict_nested_loso",
            )
            self.assertEqual(
                config["evaluation"]["internal"]["primary"]["group_unit"],
                "participant",
            )
            self.assertEqual(
                config["evaluation"]["external"]["within_cohort"]["group_unit"],
                "subject",
            )
            self.assertEqual(
                len(config["eeg"]["internal_channels"]),
                8,
            )

            for value in config["paths"].values():
                self.assertTrue(Path(value).is_absolute())


if __name__ == "__main__":
    unittest.main()
