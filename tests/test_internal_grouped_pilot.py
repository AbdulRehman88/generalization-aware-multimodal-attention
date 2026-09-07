from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.evaluation.internal_grouped_pilot import (
    build_grouped_splits,
    fit_training_cleanup,
    normalize_shap_importance,
)


class InternalGroupedPilotTests(
    unittest.TestCase
):
    def test_training_cleanup_removes_constants_and_duplicates(
        self,
    ) -> None:
        frame = pd.DataFrame(
            {
                "useful": [0.0, 1.0, 2.0, 3.0],
                "duplicate": [0.0, 1.0, 2.0, 3.0],
                "constant": [5.0, 5.0, 5.0, 5.0],
                "other": [1.0, 0.0, 1.0, 0.0],
            }
        )

        retained, summary = fit_training_cleanup(
            frame
        )

        self.assertEqual(
            retained,
            ["useful", "other"],
        )

        self.assertEqual(
            summary["constant_removed"],
            1,
        )

        self.assertEqual(
            summary["duplicate_removed"],
            1,
        )

    def test_grouped_splits_are_disjoint_and_complete(
        self,
    ) -> None:
        groups = []
        labels = []

        for participant in range(10):
            for label in (0, 1, 2):
                for _ in range(3):
                    groups.append(
                        f"P{participant:02d}"
                    )
                    labels.append(label)

        groups_array = np.asarray(groups)
        labels_array = np.asarray(labels)

        splits = build_grouped_splits(
            labels_array,
            groups_array,
            folds=5,
            random_seed=42,
        )

        test_count = np.zeros(
            len(labels_array),
            dtype=int,
        )

        for train_indices, test_indices in splits:
            self.assertTrue(
                set(groups_array[train_indices])
                .isdisjoint(
                    set(
                        groups_array[
                            test_indices
                        ]
                    )
                )
            )

            test_count[test_indices] += 1

        np.testing.assert_array_equal(
            test_count,
            np.ones(
                len(labels_array),
                dtype=int,
            ),
        )

    def test_multiclass_shap_shape_is_normalized(
        self,
    ) -> None:
        values = np.ones(
            (12, 5, 3),
            dtype=float,
        )

        importance = normalize_shap_importance(
            values,
            feature_count=5,
        )

        self.assertEqual(
            importance.shape,
            (5,),
        )

        np.testing.assert_allclose(
            importance,
            np.ones(5),
        )


if __name__ == "__main__":
    unittest.main()