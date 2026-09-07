from __future__ import annotations

import unittest

import pandas as pd

from src.evaluation.ds003838_nested_loso import (
    choose_candidate,
    validate_disjoint_roles,
)


class DS003838NestedLOSOTests(unittest.TestCase):
    def test_roles_are_complete_and_disjoint(self) -> None:
        validate_disjoint_roles(
            outer_test_participant="p4",
            inner_training_participants=[
                "p1",
                "p2",
            ],
            inner_validation_participants=[
                "p3",
            ],
            all_participants=[
                "p1",
                "p2",
                "p3",
                "p4",
            ],
        )

    def test_outer_overlap_is_rejected(self) -> None:
        with self.assertRaises(RuntimeError):
            validate_disjoint_roles(
                outer_test_participant="p4",
                inner_training_participants=[
                    "p1",
                    "p4",
                ],
                inner_validation_participants=[
                    "p2",
                    "p3",
                ],
                all_participants=[
                    "p1",
                    "p2",
                    "p3",
                    "p4",
                ],
            )

    def test_candidate_selection_uses_locked_tie_breaks(self) -> None:
        candidates = pd.DataFrame(
            [
                {
                    "candidate_index": 1,
                    "top_k_mode": "80",
                    "selected_feature_count": 80,
                    "model": "xgboost",
                    "validation_balanced_accuracy": 0.75,
                    "validation_macro_f1": 0.74,
                },
                {
                    "candidate_index": 2,
                    "top_k_mode": "20",
                    "selected_feature_count": 20,
                    "model": "extra_trees",
                    "validation_balanced_accuracy": 0.75,
                    "validation_macro_f1": 0.74,
                },
            ]
        )

        selected = choose_candidate(
            candidates,
            primary_metric="balanced_accuracy",
            secondary_metric="macro_f1",
        )

        self.assertEqual(
            selected["top_k_mode"],
            "20",
        )

        self.assertEqual(
            selected["model"],
            "extra_trees",
        )


if __name__ == "__main__":
    unittest.main()