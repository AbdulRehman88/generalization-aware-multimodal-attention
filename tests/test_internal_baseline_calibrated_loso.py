from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.evaluation.internal_baseline_calibrated_loso import (
    apply_low_class_multiplier,
    baseline_calibrate_duration,
)


class BaselineCalibratedLOSOTests(
    unittest.TestCase
):
    def test_baseline_is_centered_and_overlap_is_excluded(
        self,
    ) -> None:
        rows = []

        for participant_offset, participant in enumerate(
            [
                "P01",
                "P02",
            ]
        ):
            for phase in (
                1,
                2,
                3,
            ):
                for index, start in enumerate(
                    range(
                        0,
                        200,
                        8,
                    )
                ):
                    rows.append(
                        {
                            "segment_id":
                                f"{participant}_{phase}_{index}",
                            "participant":
                                participant,
                            "phase":
                                phase,
                            "label":
                                phase - 1,
                            "long_window_index":
                                index,
                            "start_seconds":
                                float(start),
                            "end_seconds":
                                float(
                                    start + 16
                                ),
                            "window_seconds":
                                16,
                            "stride_seconds":
                                8,
                            "feature_a":
                                (
                                    participant_offset
                                    * 100
                                    + phase * 10
                                    + index
                                ),
                            "feature_b":
                                (
                                    participant_offset
                                    * 50
                                    + phase
                                    + index * 2
                                ),
                        }
                    )

        calibrated, audit = (
            baseline_calibrate_duration(
                pd.DataFrame(rows),
                [
                    "feature_a",
                    "feature_b",
                ],
                calibration_seconds=120,
            )
        )

        self.assertEqual(
            len(audit),
            2,
        )

        for participant in (
            "P01",
            "P02",
        ):
            baseline = calibrated.loc[
                (
                    calibrated[
                        "participant"
                    ]
                    == participant
                )
                & (
                    calibrated[
                        "phase"
                    ]
                    == 1
                )
                & (
                    calibrated[
                        "end_seconds"
                    ]
                    <= 120
                ),
                [
                    "feature_a",
                    "feature_b",
                ],
            ]

            np.testing.assert_allclose(
                np.median(
                    baseline.to_numpy(
                        dtype=float
                    ),
                    axis=0,
                ),
                np.zeros(2),
                atol=1e-12,
            )

        overlapping_low = calibrated.loc[
            (
                calibrated[
                    "phase"
                ]
                == 1
            )
            & (
                calibrated[
                    "start_seconds"
                ]
                < 120
            )
        ]

        self.assertFalse(
            overlapping_low[
                "protocol_eligible"
            ].any()
        )

        later_low = calibrated.loc[
            (
                calibrated[
                    "phase"
                ]
                == 1
            )
            & (
                calibrated[
                    "start_seconds"
                ]
                >= 120
            )
        ]

        self.assertTrue(
            later_low[
                "protocol_eligible"
            ].all()
        )

    def test_low_class_multiplier_is_normalized(
        self,
    ) -> None:
        probabilities = np.asarray(
            [
                [
                    0.30,
                    0.45,
                    0.25,
                ],
            ]
        )

        adjusted = apply_low_class_multiplier(
            probabilities,
            1.5,
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