from __future__ import annotations

import unittest

import numpy as np

from src.features.internal_xr_revision_features import (
    extract_ecg_features,
    extract_eeg_features,
    extract_pupil_features,
    signal_shape_features,
)


class InternalXRRevisionFeatureTests(
    unittest.TestCase
):
    def test_constant_signal_features_are_finite(
        self,
    ) -> None:
        features = signal_shape_features(
            np.ones(512),
            sampling_rate_hz=128.0,
            nperseg=256,
        )

        self.assertTrue(
            all(
                np.isfinite(value)
                for value in features.values()
            )
        )

        self.assertEqual(
            features["skew"],
            0.0,
        )

        self.assertEqual(
            features["kurtosis"],
            0.0,
        )

    def test_eeg_schema_excludes_invalid_features(
        self,
    ) -> None:
        time = np.arange(
            512,
            dtype=float,
        ) / 128.0

        window = np.column_stack(
            [
                np.sin(
                    2
                    * np.pi
                    * frequency
                    * time
                )
                for frequency in (
                    2,
                    5,
                    9,
                    11,
                    15,
                    20,
                    32,
                    35,
                )
            ]
        )

        channels = [
            "AF7",
            "Fp1",
            "Fpz",
            "Fp2",
            "AF8",
            "O1",
            "POz",
            "O2",
        ]

        bands = {
            "delta": [0.5, 4.0],
            "theta": [4.0, 8.0],
            "alpha": [8.0, 12.0],
            "beta": [12.0, 30.0],
            "gamma_30_40": [30.0, 40.0],
        }

        features = extract_eeg_features(
            window,
            128.0,
            channels,
            bands,
            256,
        )

        joined = " ".join(
            features
        ).casefold()

        self.assertNotIn(
            "high_gamma",
            joined,
        )

        self.assertNotIn(
            "pde",
            joined,
        )

        self.assertIn(
            "gamma_30_40",
            joined,
        )

        self.assertTrue(
            all(
                np.isfinite(value)
                for value in features.values()
            )
        )

    def test_ecg_schema_contains_no_hrv_claims(
        self,
    ) -> None:
        time = np.arange(
            512,
            dtype=float,
        ) / 128.0

        signal = (
            np.sin(
                2
                * np.pi
                * 1.2
                * time
            )
        )

        features = extract_ecg_features(
            signal,
            128.0,
            256,
        )

        joined = " ".join(
            features
        ).casefold()

        self.assertNotIn(
            "hrv",
            joined,
        )

        self.assertNotIn(
            "lf_hf",
            joined,
        )

        self.assertTrue(
            all(
                np.isfinite(value)
                for value in features.values()
            )
        )

    def test_pupil_schema_excludes_absolute_gaze_means(
        self,
    ) -> None:
        time = np.arange(
            120,
            dtype=float,
        ) / 30.0

        window = np.column_stack(
            [
                50 + 2 * np.sin(time),
                60 + 3 * np.cos(time),
                30 + 0.5 * np.sin(2 * time),
            ]
        )

        features = extract_pupil_features(
            window,
            30.0,
            {
                "slow": [0.25, 1.0],
                "fast": [1.0, 4.0],
            },
            256,
        )

        self.assertNotIn(
            "gaze_x_mean",
            features,
        )

        self.assertNotIn(
            "gaze_y_mean",
            features,
        )

        self.assertTrue(
            all(
                np.isfinite(value)
                for value in features.values()
            )
        )


if __name__ == "__main__":
    unittest.main()