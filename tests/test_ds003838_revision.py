from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.preprocessing.ds003838_revision import (
    filter_notch_and_resample,
    interpolate_pupil_window,
    rational_resampling_ratio,
)


class DS003838RevisionPreprocessingTests(unittest.TestCase):
    def test_1000_to_128_ratio(self) -> None:
        self.assertEqual(
            rational_resampling_ratio(
                1000.0,
                128.0,
            ),
            (16, 125),
        )

    def test_filter_and_resample_shape(self) -> None:
        rate = 1000.0
        time = np.arange(
            14000,
            dtype=float,
        ) / rate

        signal = np.column_stack(
            [
                np.sin(
                    2.0
                    * np.pi
                    * 10.0
                    * time
                ),
                np.sin(
                    2.0
                    * np.pi
                    * 2.0
                    * time
                ),
            ]
        )

        result = filter_notch_and_resample(
            signal,
            original_rate_hz=1000.0,
            target_rate_hz=128.0,
            lowcut_hz=0.5,
            highcut_hz=40.0,
            filter_order=5,
            notch_frequency_hz=50.0,
            notch_quality_factor=30.0,
        )

        self.assertEqual(
            result.shape,
            (1792, 2),
        )

        self.assertTrue(
            np.isfinite(result).all()
        )

    def synthetic_streams(
        self,
    ) -> dict[str, pd.DataFrame]:
        pupil_times = np.arange(
            100.0,
            104.0,
            1.0 / 125.0,
        )

        gaze_times = np.arange(
            100.0,
            104.0,
            1.0 / 250.0,
        )

        pupil_stream = pd.DataFrame(
            {
                "pupil_timestamp": pupil_times,
                "diameter_3d": (
                    4.0
                    + 0.1
                    * np.sin(pupil_times)
                ),
                "confidence": np.full(
                    len(pupil_times),
                    0.95,
                ),
                "blink": np.zeros(
                    len(pupil_times)
                ),
                "eye_id": np.resize(
                    [0.0, 1.0],
                    len(pupil_times),
                ),
            }
        )

        gaze_stream = pd.DataFrame(
            {
                "gaze_timestamp": gaze_times,
                "gaze_norm_pos_x": (
                    0.5
                    + 0.01
                    * np.sin(gaze_times)
                ),
                "gaze_norm_pos_y": (
                    0.5
                    + 0.01
                    * np.cos(gaze_times)
                ),
                "confidence": np.full(
                    len(gaze_times),
                    0.95,
                ),
                "blink": np.zeros(
                    len(gaze_times)
                ),
            }
        )

        return {
            "pupil": pupil_stream,
            "gaze": gaze_stream,
        }

    def test_separate_stream_interpolation(
        self,
    ) -> None:
        window, quality = interpolate_pupil_window(
            self.synthetic_streams(),
            window_start_seconds=100.0,
            window_stop_seconds=104.0,
            timestamp_field="pupil_timestamp",
            gaze_timestamp_field="gaze_timestamp",
            gaze_x_field="gaze_norm_pos_x",
            gaze_y_field="gaze_norm_pos_y",
            diameter_field="diameter_3d",
            target_rate_hz=30.0,
            minimum_confidence=0.8,
            minimum_valid_fraction=0.8,
            maximum_gap_seconds=0.5,
        )

        self.assertIsNotNone(window)
        self.assertEqual(
            window.shape,
            (120, 3),
        )
        self.assertTrue(
            np.isfinite(window).all()
        )
        self.assertTrue(
            quality["pupil_available"]
        )
        self.assertGreater(
            quality["pupil_valid_fraction"],
            0.99,
        )
        self.assertGreater(
            quality["gaze_valid_fraction"],
            0.99,
        )

    def test_gaze_long_gap_is_rejected(
        self,
    ) -> None:
        streams = self.synthetic_streams()

        gaze = streams["gaze"]

        streams["gaze"] = gaze.loc[
            ~(
                (
                    gaze["gaze_timestamp"]
                    >= 101.0
                )
                & (
                    gaze["gaze_timestamp"]
                    < 102.0
                )
            )
        ].reset_index(drop=True)

        window, quality = interpolate_pupil_window(
            streams,
            window_start_seconds=100.0,
            window_stop_seconds=104.0,
            timestamp_field="pupil_timestamp",
            gaze_timestamp_field="gaze_timestamp",
            gaze_x_field="gaze_norm_pos_x",
            gaze_y_field="gaze_norm_pos_y",
            diameter_field="diameter_3d",
            target_rate_hz=30.0,
            minimum_confidence=0.8,
            minimum_valid_fraction=0.5,
            maximum_gap_seconds=0.5,
        )

        self.assertIsNone(window)
        self.assertFalse(
            quality["pupil_available"]
        )
        self.assertGreater(
            quality[
                "gaze_maximum_gap_seconds"
            ],
            0.5,
        )


if __name__ == "__main__":
    unittest.main()
