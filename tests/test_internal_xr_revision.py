from __future__ import annotations

import unittest

import numpy as np

from src.preprocessing.internal_xr_revision import (
    filter_and_resample,
    interpolate_short_tracking_gaps,
    parse_recording_filename,
    pupil_window_available,
    run_length_by_sample,
)


class InternalXRRevisionPreprocessingTests(
    unittest.TestCase
):
    def test_filename_parser(self) -> None:
        self.assertEqual(
            parse_recording_filename(
                "2.P13_Pupil.csv"
            ),
            (2, "P13", "Pupil"),
        )

        self.assertEqual(
            parse_recording_filename(
                "1.P01_EEG.csv"
            ),
            (1, "P01", "EEG"),
        )

    def test_short_internal_gap_is_interpolated(
        self,
    ) -> None:
        values = np.asarray(
            [
                [1.0, 10.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [4.0, 40.0],
            ]
        )

        invalid = np.asarray(
            [False, True, True, False]
        )

        cleaned = interpolate_short_tracking_gaps(
            values,
            invalid,
            maximum_gap_samples=2,
        )

        np.testing.assert_allclose(
            cleaned,
            np.asarray(
                [
                    [1.0, 10.0],
                    [2.0, 20.0],
                    [3.0, 30.0],
                    [4.0, 40.0],
                ]
            ),
        )

    def test_long_gap_remains_missing(
        self,
    ) -> None:
        values = np.asarray(
            [
                [1.0],
                [0.0],
                [0.0],
                [0.0],
                [5.0],
            ]
        )

        invalid = np.asarray(
            [False, True, True, True, False]
        )

        cleaned = interpolate_short_tracking_gaps(
            values,
            invalid,
            maximum_gap_samples=2,
        )

        self.assertTrue(
            np.isnan(
                cleaned[1:4]
            ).all()
        )

    def test_locked_pupil_quality_rule(
        self,
    ) -> None:
        invalid = np.zeros(
            120,
            dtype=bool,
        )

        invalid[20:30] = True

        runs = run_length_by_sample(
            invalid
        )

        self.assertTrue(
            pupil_window_available(
                invalid,
                runs,
                minimum_valid_fraction=0.80,
                maximum_gap_samples=15,
            )
        )

        invalid[20:40] = True

        runs = run_length_by_sample(
            invalid
        )

        self.assertFalse(
            pupil_window_available(
                invalid,
                runs,
                minimum_valid_fraction=0.80,
                maximum_gap_samples=15,
            )
        )

    def test_filter_and_resample_shape(
        self,
    ) -> None:
        time = np.arange(
            512 * 4,
            dtype=float,
        ) / 512.0

        signal = (
            np.sin(
                2.0
                * np.pi
                * 10.0
                * time
            )
        )[:, np.newaxis]

        processed = filter_and_resample(
            signal,
            original_rate_hz=512.0,
            target_rate_hz=128.0,
            lowcut_hz=0.5,
            highcut_hz=40.0,
            filter_order=5,
        )

        self.assertEqual(
            processed.shape,
            (128 * 4, 1),
        )

        self.assertTrue(
            np.isfinite(
                processed
            ).all()
        )


if __name__ == "__main__":
    unittest.main()