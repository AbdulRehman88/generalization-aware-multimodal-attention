from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.evaluation.internal_nested_loso_long_windows import (
    causal_probability_smoothing,
    deterministic_inner_split,
    history_windows_for_duration,
    selected_features_for_mode,
)


class NestedLOSOLongWindowTests(
    unittest.TestCase
):
    def test_inner_split_excludes_outer_participant(
        self,
    ) -> None:
        participants = [
            f"P{index:02d}"
            for index in range(
                1,
                14,
            )
        ]

        training, validation = (
            deterministic_inner_split(
                participants,
                "P05",
                validation_count=2,
            )
        )

        self.assertEqual(
            len(training),
            10,
        )

        self.assertEqual(
            len(validation),
            2,
        )

        self.assertNotIn(
            "P05",
            training,
        )

        self.assertNotIn(
            "P05",
            validation,
        )

        self.assertFalse(
            set(training).intersection(
                validation
            )
        )

    def test_history_conversion_rounds_up(
        self,
    ) -> None:
        windows, seconds = (
            history_windows_for_duration(
                60,
                8,
            )
        )

        self.assertEqual(
            windows,
            8,
        )

        self.assertEqual(
            seconds,
            64,
        )

        windows, seconds = (
            history_windows_for_duration(
                0,
                8,
            )
        )

        self.assertEqual(
            windows,
            1,
        )

        self.assertEqual(
            seconds,
            0,
        )

    def test_causal_smoothing_does_not_cross_phase(
        self,
    ) -> None:
        metadata = pd.DataFrame(
            {
                "participant": [
                    "P01",
                    "P01",
                    "P01",
                    "P01",
                ],
                "phase": [
                    1,
                    1,
                    2,
                    2,
                ],
                "long_window_index": [
                    0,
                    1,
                    0,
                    1,
                ],
            }
        )

        probabilities = np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
            ]
        )

        smoothed = causal_probability_smoothing(
            metadata,
            probabilities,
            smoothing_windows=2,
        )

        np.testing.assert_allclose(
            smoothed[0],
            [1.0, 0.0, 0.0],
        )

        np.testing.assert_allclose(
            smoothed[1],
            [0.5, 0.5, 0.0],
        )

        np.testing.assert_allclose(
            smoothed[2],
            [0.0, 0.0, 1.0],
        )

        np.testing.assert_allclose(
            smoothed[3],
            [0.5, 0.0, 0.5],
        )

    def test_top_k_and_all_modes(
        self,
    ) -> None:
        retained = [
            "a",
            "b",
            "c",
        ]

        ranked = [
            "c",
            "a",
            "b",
        ]

        self.assertEqual(
            selected_features_for_mode(
                "2",
                retained,
                ranked,
            ),
            [
                "c",
                "a",
            ],
        )

        self.assertEqual(
            selected_features_for_mode(
                "all",
                retained,
                ranked,
            ),
            retained,
        )


if __name__ == "__main__":
    unittest.main()