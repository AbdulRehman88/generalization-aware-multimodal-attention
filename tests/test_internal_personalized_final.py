from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.evaluation.internal_personalized_final import (
    apply_low_class_multiplier,
    assign_final_roles,
    causal_smooth_probabilities,
    compute_prototype_distances,
    distances_to_probabilities,
    fit_prototype_space,
)


class PersonalizedFinalProtocolTests(
    unittest.TestCase
):
    def test_role_assignment_preserves_untouched_test(
        self,
    ) -> None:
        context = pd.DataFrame(
            {
                "window_index":
                    np.arange(4, 75),
            }
        )

        roles = assign_final_roles(
            context,
            calibration_windows=15,
            first_guard_windows=5,
            validation_windows=15,
            second_guard_windows=5,
        )

        self.assertEqual(
            int(
                (
                    roles
                    == "calibration"
                ).sum()
            ),
            11,
        )

        self.assertEqual(
            int(
                (
                    roles
                    == "guard_1"
                ).sum()
            ),
            5,
        )

        self.assertEqual(
            int(
                (
                    roles
                    == "validation"
                ).sum()
            ),
            15,
        )

        self.assertEqual(
            int(
                (
                    roles
                    == "guard_2"
                ).sum()
            ),
            5,
        )

        self.assertEqual(
            int(
                (
                    roles
                    == "test"
                ).sum()
            ),
            35,
        )

    def test_prototype_probability_favors_nearest_class(
        self,
    ) -> None:
        adaptation = pd.DataFrame(
            {
                "participant": [
                    "P01",
                    "P01",
                    "P01",
                    "P01",
                    "P01",
                    "P01",
                ],
                "label": [
                    0,
                    0,
                    1,
                    1,
                    2,
                    2,
                ],
                "feature_a": [
                    0.0,
                    0.1,
                    5.0,
                    5.1,
                    10.0,
                    10.1,
                ],
                "feature_b": [
                    0.0,
                    0.2,
                    5.0,
                    5.2,
                    10.0,
                    10.2,
                ],
            }
        )

        target = pd.DataFrame(
            {
                "participant": [
                    "P01",
                ],
                "feature_a": [
                    5.05,
                ],
                "feature_b": [
                    5.05,
                ],
            }
        )

        prototype_space = fit_prototype_space(
            adaptation,
            [
                "feature_a",
                "feature_b",
            ],
        )

        distances = compute_prototype_distances(
            prototype_space,
            target,
            [
                "feature_a",
                "feature_b",
            ],
        )

        probabilities = distances_to_probabilities(
            distances,
            temperature=1.0,
        )

        self.assertEqual(
            int(
                np.argmax(
                    probabilities[0]
                )
            ),
            1,
        )

        np.testing.assert_allclose(
            probabilities.sum(axis=1),
            np.ones(1),
        )

    def test_causal_smoothing_uses_no_future_rows(
        self,
    ) -> None:
        metadata = pd.DataFrame(
            {
                "participant": [
                    "P01",
                    "P01",
                    "P01",
                ],
                "phase": [
                    1,
                    1,
                    1,
                ],
                "window_index": [
                    0,
                    1,
                    2,
                ],
            }
        )

        probabilities = np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )

        smoothed = causal_smooth_probabilities(
            metadata,
            probabilities,
            smoothing_windows=2,
        )

        np.testing.assert_allclose(
            smoothed[0],
            probabilities[0],
        )

        np.testing.assert_allclose(
            smoothed[1],
            np.asarray(
                [0.5, 0.5, 0.0]
            ),
        )

        np.testing.assert_allclose(
            smoothed[2],
            np.asarray(
                [0.0, 0.5, 0.5]
            ),
        )

    def test_low_class_multiplier_renormalizes(
        self,
    ) -> None:
        probabilities = np.asarray(
            [
                [0.3, 0.4, 0.3],
            ]
        )

        adjusted = apply_low_class_multiplier(
            probabilities,
            multiplier=1.5,
        )

        self.assertGreater(
            adjusted[0, 0],
            probabilities[0, 0],
        )

        np.testing.assert_allclose(
            adjusted.sum(axis=1),
            np.ones(1),
        )


if __name__ == "__main__":
    unittest.main()