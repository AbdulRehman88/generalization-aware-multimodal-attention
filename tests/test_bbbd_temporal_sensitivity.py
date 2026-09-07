from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import yaml

from src.features.bbbd_temporal_sensitivity import (
    EXPECTED_DURATIONS,
    build_duration_config,
    duration_directory_name,
    run_temporal_sensitivity_features,
    validate_contract,
)


class BBBDTemporalSensitivityFeatureTests(unittest.TestCase):

    def primary_config(self) -> dict:
        return {
            "windowing": {
                "duration_seconds": 4.0,
                "overlap_fraction": 0.5,
                "edge_trim_seconds": 2.0,
            },
            "outputs": {
                "full_directory": "unused",
                "smoke_directory": "unused",
            },
        }

    def contract(self) -> dict:
        return {
            "primary_real_time_reference": {
                "duration_seconds": 4,
                "overlap_fraction": 0.5,
                "stride_seconds": 2,
                "may_be_replaced_by_sensitivity": False,
            },
            "sensitivity_windows": [
                {
                    "duration_seconds": 8,
                    "overlap_fraction": 0.5,
                    "stride_seconds": 4,
                },
                {
                    "duration_seconds": 16,
                    "overlap_fraction": 0.5,
                    "stride_seconds": 8,
                },
                {
                    "duration_seconds": 24,
                    "overlap_fraction": 0.5,
                    "stride_seconds": 12,
                },
                {
                    "duration_seconds": 32,
                    "overlap_fraction": 0.5,
                    "stride_seconds": 16,
                },
            ],
            "feature_generation": {
                "recompute_features_for_each_duration": True,
                "average_existing_four_second_feature_rows": False,
                "concatenate_existing_feature_vectors": False,
            },
        }

    def test_duration_directory_name_is_deterministic(self) -> None:
        self.assertEqual(
            duration_directory_name(
                8,
                0.5,
            ),
            "window_08s_overlap_50pct",
        )

        self.assertEqual(
            duration_directory_name(
                32,
                0.5,
            ),
            "window_32s_overlap_50pct",
        )

    def test_contract_grid_is_exact(self) -> None:
        windows = validate_contract(
            self.primary_config(),
            self.contract(),
        )

        self.assertEqual(
            [
                row["duration_seconds"]
                for row in windows
            ],
            list(EXPECTED_DURATIONS),
        )

    def test_duration_config_changes_window_only(self) -> None:
        primary = self.primary_config()

        generated = build_duration_config(
            primary,
            16,
            0.5,
            "output/window16",
        )

        self.assertEqual(
            primary[
                "windowing"
            ][
                "duration_seconds"
            ],
            4.0,
        )

        self.assertEqual(
            generated[
                "windowing"
            ][
                "duration_seconds"
            ],
            16.0,
        )

        self.assertEqual(
            generated[
                "windowing"
            ][
                "overlap_fraction"
            ],
            0.5,
        )

        self.assertEqual(
            generated[
                "outputs"
            ][
                "full_directory"
            ],
            "output/window16",
        )

    def test_unknown_duration_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            primary_path = root / "primary.yaml"
            contract_path = root / "contract.yaml"

            primary_path.write_text(
                yaml.safe_dump(
                    self.primary_config()
                ),
                encoding="utf-8",
            )

            contract_path.write_text(
                yaml.safe_dump(
                    self.contract()
                ),
                encoding="utf-8",
            )

            with self.assertRaises(
                ValueError
            ):
                run_temporal_sensitivity_features(
                    primary_path,
                    contract_path,
                    root / "result",
                    smoke=True,
                    durations=[12],
                )

    def test_wrapper_orchestrates_all_durations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            primary_path = root / "primary.yaml"
            contract_path = root / "contract.yaml"
            output_root = root / "result"

            primary_path.write_text(
                yaml.safe_dump(
                    self.primary_config()
                ),
                encoding="utf-8",
            )

            contract_path.write_text(
                yaml.safe_dump(
                    self.contract()
                ),
                encoding="utf-8",
            )

            calls: list[int] = []

            def fake_builder(
                config_path: str | Path,
                output_directory: str | Path,
                smoke: bool,
            ) -> None:
                config = yaml.safe_load(
                    Path(config_path).read_text(
                        encoding="utf-8"
                    )
                )

                duration = int(
                    config[
                        "windowing"
                    ][
                        "duration_seconds"
                    ]
                )

                calls.append(
                    duration
                )

                output = Path(
                    output_directory
                )

                output.mkdir(
                    parents=True,
                    exist_ok=False,
                )

                summary = {
                    "requested_recordings":
                        6,
                    "completed_recordings":
                        6,
                    "candidate_windows":
                        100,
                    "accepted_windows":
                        90,
                    "rejected_pupil_windows":
                        10,
                    "rejected_nonfinite_feature_windows":
                        0,
                    "window_duration_seconds":
                        float(duration),
                    "window_overlap_fraction":
                        0.5,
                    "feature_selection_performed":
                        False,
                }

                (
                    output
                    / "summary.json"
                ).write_text(
                    json.dumps(
                        summary
                    ),
                    encoding="utf-8",
                )

            with patch(
                "src.features."
                "bbbd_temporal_sensitivity."
                "build_bbbd_features",
                side_effect=fake_builder,
            ):
                manifest = (
                    run_temporal_sensitivity_features(
                        primary_path,
                        contract_path,
                        output_root,
                        smoke=True,
                    )
                )

            self.assertEqual(
                calls,
                [8, 16, 24, 32],
            )

            self.assertEqual(
                [
                    row[
                        "duration_seconds"
                    ]
                    for row in manifest[
                        "durations"
                    ]
                ],
                [8, 16, 24, 32],
            )

            self.assertTrue(
                (
                    output_root
                    / "temporal_sensitivity_manifest.json"
                ).is_file()
            )

            self.assertFalse(
                output_root.with_name(
                    output_root.name
                    + ".partial"
                ).exists()
            )

    def test_existing_output_is_not_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            primary_path = root / "primary.yaml"
            contract_path = root / "contract.yaml"
            output_root = root / "existing"

            primary_path.write_text(
                yaml.safe_dump(
                    self.primary_config()
                ),
                encoding="utf-8",
            )

            contract_path.write_text(
                yaml.safe_dump(
                    self.contract()
                ),
                encoding="utf-8",
            )

            output_root.mkdir()

            with self.assertRaises(
                FileExistsError
            ):
                run_temporal_sensitivity_features(
                    primary_path,
                    contract_path,
                    output_root,
                    smoke=True,
                )


if __name__ == "__main__":
    unittest.main()
