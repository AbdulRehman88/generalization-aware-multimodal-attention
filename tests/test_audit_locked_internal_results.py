from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.evaluation.audit_locked_internal_results import (
    cluster_bootstrap,
    compare_metrics,
    validate_outer_structure,
    validate_probabilities,
)


class LockedInternalResultsAuditTests(unittest.TestCase):
    def synthetic_predictions(self) -> pd.DataFrame:
        rows = []

        for fold, participant in enumerate(
            ["P01", "P02", "P03"],
            start=1,
        ):
            for label in [0, 1, 2]:
                probabilities = [0.05, 0.05, 0.05]
                probabilities[label] = 0.90

                rows.append(
                    {
                        "segment_id": f"{participant}_{label}",
                        "participant": participant,
                        "outer_fold": fold,
                        "true_label": label,
                        "predicted_label": label,
                        "probability_class_0": probabilities[0],
                        "probability_class_1": probabilities[1],
                        "probability_class_2": probabilities[2],
                    }
                )

        return pd.DataFrame(rows)

    def test_probability_validation(self) -> None:
        validate_probabilities(
            np.asarray(
                [
                    [0.2, 0.3, 0.5],
                    [0.1, 0.1, 0.8],
                ]
            )
        )

        with self.assertRaises(RuntimeError):
            validate_probabilities(
                np.asarray([[0.2, 0.3, 0.6]])
            )

    def test_cluster_bootstrap_is_deterministic(self) -> None:
        frame = self.synthetic_predictions()

        _, first = cluster_bootstrap(
            frame,
            repetitions=20,
            random_seed=42,
        )

        _, second = cluster_bootstrap(
            frame,
            repetitions=20,
            random_seed=42,
        )

        pd.testing.assert_frame_equal(first, second)


    def test_metric_comparison_handles_csv_probability_precision(
        self,
    ) -> None:
        expected = {
            "accuracy": 0.80,
            "balanced_accuracy": 0.79,
            "macro_f1": 0.78,
            "macro_roc_auc_ovr": 0.9000000,
            "macro_pr_auc": 0.8500000,
        }

        observed = {
            **expected,
            "macro_roc_auc_ovr": 0.9000011,
            "macro_pr_auc": 0.8499989,
        }

        compare_metrics(
            observed,
            expected,
            "synthetic",
        )

        materially_changed = {
            **expected,
            "macro_roc_auc_ovr": 0.9001,
        }

        with self.assertRaises(RuntimeError):
            compare_metrics(
                materially_changed,
                expected,
                "synthetic",
            )

    def test_outer_structure_rejects_duplicate_segments(self) -> None:
        frame = self.synthetic_predictions()
        frame.loc[1, "segment_id"] = frame.loc[0, "segment_id"]

        with self.assertRaises(RuntimeError):
            validate_outer_structure(frame, "synthetic")


if __name__ == "__main__":
    unittest.main()