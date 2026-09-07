from __future__ import annotations

import json
import unittest
from pathlib import Path

import pandas as pd

from src.evaluation.bbbd_temporal_matched_preflight import (
    normalize_cleanup_result,
    unwrap_loaded_path_table,
)


JSON_PATH = Path(
    "_research_audit/"
    "bbbd_temporal_matched_preflight_v1.json"
)

CSV_PATH = Path(
    "_research_audit/"
    "bbbd_temporal_matched_preflight_v1.csv"
)


class BBBDTemporalMatchedPreflightTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.audit = json.loads(
            JSON_PATH.read_text(
                encoding="utf-8"
            )
        )

        cls.summary = pd.read_csv(
            CSV_PATH
        )

    def test_count_backed_cleanup_evidence_is_accepted(self) -> None:
        observed = normalize_cleanup_result(
            (
                ["Pupil__mean"],
                {
                    "constant_feature_count": 0,
                    "duplicate_feature_count": 2,
                },
            ),
            [
                "Pupil__mean",
                "Pupil__median",
                "Pupil__range",
            ],
        )

        self.assertEqual(
            observed["retained_feature_count"],
            1,
        )

        self.assertEqual(
            observed["constant_feature_count"],
            0,
        )

        self.assertEqual(
            observed["duplicate_feature_count"],
            2,
        )

        self.assertEqual(
            observed["unnamed_removed_features"],
            [
                "Pupil__median",
                "Pupil__range",
            ],
        )

    def test_missing_cleanup_subtype_metadata_is_not_invented(self) -> None:
        observed = normalize_cleanup_result(
            (
                ["Pupil__mean"],
            ),
            [
                "Pupil__mean",
                "Pupil__median",
                "Pupil__range",
            ],
        )

        self.assertEqual(
            observed["original_feature_count"],
            3,
        )

        self.assertEqual(
            observed["retained_feature_count"],
            1,
        )

        self.assertEqual(
            observed["removed_feature_count"],
            2,
        )

        self.assertEqual(
            observed["constant_feature_count"],
            0,
        )

        self.assertEqual(
            observed["duplicate_feature_count"],
            0,
        )

        self.assertEqual(
            observed["unclassified_removed_feature_count"],
            2,
        )

        self.assertFalse(
            observed["cleanup_subtype_breakdown_complete"]
        )

    def test_path_loader_tuple_is_unwrapped(self) -> None:
        frame = pd.DataFrame(
            {
                "segment_id": ["s1"],
                "value": [1.0],
            }
        )

        observed = unwrap_loaded_path_table(
            (
                frame,
                ["value"],
            )
        )

        self.assertIs(
            observed,
            frame,
        )

        with self.assertRaises(TypeError):
            unwrap_loaded_path_table(
                (
                    ["value"],
                    {"metadata": True},
                )
            )

    def test_preflight_passed_without_fitting(self) -> None:
        self.assertEqual(
            self.audit["status"],
            "passed",
        )

        self.assertFalse(
            self.audit["performance_observed"]
        )

        self.assertFalse(
            self.audit["feature_selector_fitted"]
        )

        self.assertFalse(
            self.audit["classifier_fitted"]
        )

        self.assertFalse(
            self.audit["threshold_selected"]
        )

    def test_exact_contract_and_fit_totals(self) -> None:
        self.assertEqual(
            self.audit[
                "path_split_contracts_checked"
            ],
            1330,
        )

        self.assertEqual(
            self.audit[
                "candidate_model_fits"
            ],
            7980,
        )

        self.assertEqual(
            self.audit[
                "shap_selector_fits"
            ],
            1330,
        )

        self.assertEqual(
            self.audit[
                "selected_candidate_final_refits"
            ],
            1330,
        )

        self.assertEqual(
            self.audit[
                "total_estimator_fits"
            ],
            10640,
        )

    def test_every_duration_and_path_was_checked(self) -> None:
        self.assertEqual(
            len(self.summary),
            35,
        )

        self.assertEqual(
            set(
                self.summary[
                    "duration_seconds"
                ].astype(int)
            ),
            {4, 8, 16, 24, 32},
        )

        self.assertTrue(
            (
                self.summary[
                    "contracts_checked"
                ].astype(int)
                == 38
            ).all()
        )

    def test_candidate_contracts_all_passed(self) -> None:
        self.assertEqual(
            self.audit[
                "candidate_count_mismatches"
            ],
            0,
        )

        self.assertTrue(
            self.audit[
                "role_contracts_passed"
            ]
        )

        self.assertTrue(
            self.audit[
                "training_only_cleanup_checked"
            ]
        )

        self.assertTrue(
            self.summary[
                "candidate_contracts_passed"
            ].astype(bool).all()
        )

    def test_primary_result_remains_unchanged(self) -> None:
        self.assertTrue(
            self.audit[
                "primary_four_second_392_recording_result_unchanged"
            ]
        )

        self.assertEqual(
            self.audit[
                "matched_four_second_role"
            ],
            "sensitivity reference only",
        )


if __name__ == "__main__":
    unittest.main()
