from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd
import yaml


class BBBDTemporalSensitivityAmendmentTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.config = yaml.safe_load(
            Path(
                "configs/"
                "bbbd_temporal_sensitivity_amendment_v1.yaml"
            ).read_text(
                encoding="utf-8"
            )
        )

        cls.registry = pd.read_csv(
            "artifacts/revision/manifests/"
            "bbbd_temporal_sensitivity_recording_registry_v1.csv"
        )

    def test_primary_four_second_analysis_is_immutable(self) -> None:
        primary = self.config["primary_analysis"]

        self.assertEqual(
            primary["window_seconds"],
            4,
        )

        self.assertEqual(
            primary["recordings"],
            392,
        )

        self.assertFalse(
            primary["replaced"]
        )

    def test_matched_sensitivity_grid_is_exact(self) -> None:
        sensitivity = self.config[
            "matched_sensitivity_analysis"
        ]

        self.assertEqual(
            sensitivity["windows_seconds"],
            [4, 8, 16, 24, 32],
        )

        self.assertEqual(
            sensitivity["participants"],
            36,
        )

        self.assertEqual(
            sensitivity["recordings"],
            387,
        )

        self.assertFalse(
            sensitivity["performance_based_selection"]
        )

    def test_registry_contains_objective_intersection(self) -> None:
        self.assertEqual(
            len(self.registry),
            392,
        )

        selected = self.registry[
            self.registry[
                "temporal_common_complete_case"
            ]
        ]

        self.assertEqual(
            len(selected),
            387,
        )

        self.assertEqual(
            selected["participant"].nunique(),
            36,
        )

        self.assertEqual(
            set(
                selected["label"].astype(int)
            ),
            {0, 1},
        )

    def test_exact_excluded_recordings_are_locked(self) -> None:
        expected = {
            "experiment3::sub-07::ses-02::stim04",
            "experiment3::sub-09::ses-01::stim06",
            "experiment3::sub-09::ses-02::stim01",
            "experiment3::sub-09::ses-02::stim03",
            "experiment3::sub-09::ses-02::stim04",
        }

        observed = set(
            self.registry.loc[
                ~self.registry[
                    "temporal_common_complete_case"
                ],
                "recording_id",
            ]
        )

        self.assertEqual(
            observed,
            expected,
        )

    def test_pupil_rules_and_claim_boundaries_are_preserved(self) -> None:
        unchanged = self.config[
            "unchanged_rules"
        ]

        prohibited = self.config[
            "prohibited"
        ]

        self.assertTrue(
            unchanged["pupil_quality_rule"]
        )

        self.assertTrue(
            unchanged["participant_cohort"]
        )

        self.assertTrue(
            unchanged["four_second_primary_result"]
        )

        self.assertTrue(
            prohibited["relax_pupil_quality"]
        )

        self.assertTrue(
            prohibited["select_recordings_using_performance"]
        )

        self.assertTrue(
            prohibited["replace_primary_four_second_result"]
        )


if __name__ == "__main__":
    unittest.main()
