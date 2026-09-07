from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.evaluation.internal_few_shot_subject_adaptive_loso import (
    normalize_by_low_state_calibration,
    split_subject_calibration_and_test,
)


class FewShotSubjectAdaptiveLOSOTests(
    unittest.TestCase
):
    def synthetic_subject(
        self,
    ) -> pd.DataFrame:
        rows = []

        for phase in (
            1,
            2,
            3,
        ):
            for index, start in enumerate(
                range(
                    0,
                    240,
                    8,
                )
            ):
                rows.append(
                    {
                        "segment_id":
                            f"P01_{phase}_{index}",
                        "participant":
                            "P01",
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
                            phase * 10 + index,
                        "feature_b":
                            phase * 5 + index * 2,
                    }
                )

        return pd.DataFrame(
            rows
        )

    def test_low_state_normalization_centers_calibration(
        self,
    ) -> None:
        frame = self.synthetic_subject()

        normalized, audit = (
            normalize_by_low_state_calibration(
                frame,
                [
                    "feature_a",
                    "feature_b",
                ],
                calibration_budget_seconds=60,
            )
        )

        self.assertEqual(
            len(audit),
            1,
        )

        baseline = normalized.loc[
            (
                normalized[
                    "phase"
                ]
                == 1
            )
            & (
                normalized[
                    "start_seconds"
                ]
                < 60
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

    def test_calibration_and_test_are_raw_disjoint(
        self,
    ) -> None:
        frame = self.synthetic_subject()

        calibration, test, audit = (
            split_subject_calibration_and_test(
                frame,
                calibration_budget_seconds=60,
                guard_window_multiples=1,
            )
        )

        self.assertEqual(
            set(
                calibration[
                    "label"
                ].astype(int)
            ),
            {
                0,
                1,
                2,
            },
        )

        self.assertEqual(
            len(audit),
            3,
        )

        for phase in (
            1,
            2,
            3,
        ):
            phase_calibration = calibration.loc[
                calibration[
                    "phase"
                ]
                == phase
            ]

            phase_test = test.loc[
                test[
                    "phase"
                ]
                == phase
            ]

            calibration_end = float(
                phase_calibration[
                    "end_seconds"
                ].max()
            )

            test_start = float(
                phase_test[
                    "start_seconds"
                ].min()
            )

            self.assertGreaterEqual(
                test_start,
                calibration_end + 16,
            )

        self.assertFalse(
            set(
                calibration[
                    "segment_id"
                ]
            ).intersection(
                set(
                    test[
                        "segment_id"
                    ]
                )
            )
        )


if __name__ == "__main__":
    unittest.main()