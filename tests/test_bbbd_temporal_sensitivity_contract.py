from __future__ import annotations

import unittest
from pathlib import Path

import yaml


class BBBDTemporalSensitivityContractTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        path = Path(
            "configs/bbbd_temporal_sensitivity.yaml"
        )

        cls.contract = yaml.safe_load(
            path.read_text(
                encoding="utf-8"
            )
        )

    def test_primary_reference_is_immutable_four_seconds(self) -> None:
        primary = self.contract[
            "primary_real_time_reference"
        ]

        self.assertEqual(
            primary["duration_seconds"],
            4,
        )

        self.assertEqual(
            primary["overlap_fraction"],
            0.5,
        )

        self.assertFalse(
            primary["may_be_replaced_by_sensitivity"]
        )

    def test_sensitivity_grid_is_exact(self) -> None:
        windows = self.contract[
            "sensitivity_windows"
        ]

        self.assertEqual(
            [
                row["duration_seconds"]
                for row in windows
            ],
            [8, 16, 24, 32],
        )

        self.assertEqual(
            [
                row["stride_seconds"]
                for row in windows
            ],
            [4, 8, 12, 16],
        )

        self.assertTrue(
            all(
                row["overlap_fraction"] == 0.5
                for row in windows
            )
        )

    def test_raw_features_must_be_recomputed(self) -> None:
        feature_contract = self.contract[
            "feature_generation"
        ]

        self.assertTrue(
            feature_contract[
                "recompute_features_for_each_duration"
            ]
        )

        self.assertFalse(
            feature_contract[
                "average_existing_four_second_feature_rows"
            ]
        )

        self.assertFalse(
            feature_contract[
                "concatenate_existing_feature_vectors"
            ]
        )

    def test_cohort_and_paths_are_locked(self) -> None:
        cohort = self.contract["cohort"]

        self.assertEqual(
            cohort["participants"],
            36,
        )

        self.assertEqual(
            cohort["recordings"],
            392,
        )

        self.assertFalse(
            cohort["cohort_change_allowed"]
        )

        self.assertEqual(
            len(
                self.contract["modality_paths"]
            ),
            7,
        )

    def test_leakage_safe_evaluation_is_required(self) -> None:
        evaluation = self.contract["evaluation"]

        self.assertTrue(
            evaluation["participant_roles_disjoint"]
        )

        self.assertTrue(
            evaluation["cleanup_training_only"]
        )

        self.assertTrue(
            evaluation["shap_training_only"]
        )

        self.assertTrue(
            evaluation[
                "model_selection_training_validation_only"
            ]
        )

        self.assertTrue(
            evaluation[
                "threshold_selection_validation_only"
            ]
        )

        self.assertTrue(
            evaluation["test_used_once"]
        )

    def test_calibration_and_smoothing_are_excluded(self) -> None:
        prohibited = self.contract[
            "prohibited_initial_variants"
        ]

        for key in [
            "subject_calibration",
            "few_shot_adaptation",
            "probability_history_smoothing",
            "personalized_prototypes",
            "raw_60_second_windows",
            "raw_120_second_windows",
            "performance_based_primary_replacement",
            "report_only_best_duration",
        ]:
            self.assertTrue(
                prohibited[key]
            )


if __name__ == "__main__":
    unittest.main()
