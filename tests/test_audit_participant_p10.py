from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.evaluation.audit_participant_p10 import (
    evaluate_hard_failures,
    percentile_rank,
    robust_location_scale,
)


class ParticipantP10AuditTests(
    unittest.TestCase
):
    def test_robust_location_scale_is_finite(
        self,
    ) -> None:
        values = np.asarray(
            [
                [1.0, 2.0, 5.0],
                [1.0, 4.0, 5.0],
                [1.0, 6.0, 5.0],
            ]
        )

        center, scale = robust_location_scale(
            values
        )

        np.testing.assert_allclose(
            center,
            [1.0, 4.0, 5.0],
        )

        self.assertTrue(
            np.isfinite(scale).all()
        )

        self.assertTrue(
            (
                scale > 0
            ).all()
        )

    def test_percentile_rank(self) -> None:
        reference = np.asarray(
            [
                1.0,
                2.0,
                3.0,
                4.0,
            ]
        )

        self.assertEqual(
            percentile_rank(
                3.0,
                reference,
            ),
            0.75,
        )

    def test_hard_failure_rules_are_objective(
        self,
    ) -> None:
        valid = pd.DataFrame(
            {
                "archive_exists":
                    [True, True, True],
                "channel_schema_correct":
                    [True, True, True],
                "window_count":
                    [75, 105, 75],
                "manifest_window_count":
                    [75, 105, 75],
                "nonfinite_eeg_samples":
                    [0, 0, 0],
                "nonfinite_ecg_samples":
                    [0, 0, 0],
                "maximum_eeg_flat_window_fraction":
                    [0.0, 0.0, 0.0],
                "ecg_flat_window_fraction":
                    [0.0, 0.0, 0.0],
            }
        )

        failures = evaluate_hard_failures(
            valid,
            expected_recording_count=3,
            maximum_flat_fraction=0.05,
        )

        self.assertEqual(
            failures,
            [],
        )

        invalid = valid.copy()

        invalid.loc[
            0,
            "nonfinite_eeg_samples",
        ] = 1

        failures = evaluate_hard_failures(
            invalid,
            expected_recording_count=3,
            maximum_flat_fraction=0.05,
        )

        self.assertIn(
            "nonfinite_eeg_samples",
            failures,
        )


if __name__ == "__main__":
    unittest.main()