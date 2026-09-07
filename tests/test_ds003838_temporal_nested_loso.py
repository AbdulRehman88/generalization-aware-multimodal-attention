from __future__ import annotations

import unittest

import pandas as pd

from src.evaluation.ds003838_temporal_nested_loso import (
    choose_temporal_candidate,
    temporal_window_sort_value,
)


class DS003838TemporalNestedLOSOTests(
    unittest.TestCase
):
    def test_locked_temporal_order(self) -> None:
        self.assertEqual(
            temporal_window_sort_value(
                "late_encoding"
            ),
            0,
        )

        self.assertEqual(
            temporal_window_sort_value(
                "immediate_retention"
            ),
            1,
        )

        self.assertEqual(
            temporal_window_sort_value(
                "delayed_retention"
            ),
            2,
        )

    def test_primary_metric_selects_best_window(self) -> None:
        candidates = pd.DataFrame(
            [
                {
                    "candidate_index": 1,
                    "temporal_window":
                        "late_encoding",
                    "top_k_mode": "20",
                    "selected_feature_count": 20,
                    "model": "extra_trees",
                    "validation_balanced_accuracy":
                        0.40,
                    "validation_macro_f1":
                        0.39,
                },
                {
                    "candidate_index": 2,
                    "temporal_window":
                        "immediate_retention",
                    "top_k_mode": "20",
                    "selected_feature_count": 20,
                    "model": "extra_trees",
                    "validation_balanced_accuracy":
                        0.45,
                    "validation_macro_f1":
                        0.41,
                },
            ]
        )

        chosen = choose_temporal_candidate(
            candidates,
            primary_metric=
                "balanced_accuracy",
            secondary_metric=
                "macro_f1",
        )

        self.assertEqual(
            chosen[
                "temporal_window"
            ],
            "immediate_retention",
        )

    def test_exact_tie_uses_prespecified_window_order(self) -> None:
        candidates = pd.DataFrame(
            [
                {
                    "candidate_index": 1,
                    "temporal_window":
                        "delayed_retention",
                    "top_k_mode": "20",
                    "selected_feature_count": 20,
                    "model": "extra_trees",
                    "validation_balanced_accuracy":
                        0.40,
                    "validation_macro_f1":
                        0.39,
                },
                {
                    "candidate_index": 2,
                    "temporal_window":
                        "late_encoding",
                    "top_k_mode": "20",
                    "selected_feature_count": 20,
                    "model": "extra_trees",
                    "validation_balanced_accuracy":
                        0.40,
                    "validation_macro_f1":
                        0.39,
                },
            ]
        )

        chosen = choose_temporal_candidate(
            candidates,
            primary_metric=
                "balanced_accuracy",
            secondary_metric=
                "macro_f1",
        )

        self.assertEqual(
            chosen[
                "temporal_window"
            ],
            "late_encoding",
        )

    def test_missing_window_column_is_rejected(self) -> None:
        candidates = pd.DataFrame(
            [
                {
                    "candidate_index": 1,
                    "top_k_mode": "20",
                    "selected_feature_count": 20,
                    "model": "extra_trees",
                    "validation_balanced_accuracy":
                        0.40,
                    "validation_macro_f1":
                        0.39,
                }
            ]
        )

        with self.assertRaises(
            ValueError
        ):
            choose_temporal_candidate(
                candidates,
                primary_metric=
                    "balanced_accuracy",
                secondary_metric=
                    "macro_f1",
            )


if __name__ == "__main__":
    unittest.main()