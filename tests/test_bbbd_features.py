from __future__ import annotations

import unittest

import numpy as np

from src.features.bbbd_features import (
    contiguous_false_runs,
    interpolate_short_invalid_runs,
    load_direct_cohort_participants,
    load_bbbd_config,
    namespaced_participant,
    preprocess_pupil,
    window_start_indices,
)


class BBBDFeatureTests(
    unittest.TestCase
):
    def test_namespaced_participant(self) -> None:
        self.assertEqual(
            namespaced_participant(
                "experiment2",
                "sub-02",
            ),
            "experiment2::sub-02",
        )

        self.assertNotEqual(
            namespaced_participant(
                "experiment2",
                "sub-02",
            ),
            namespaced_participant(
                "experiment3",
                "sub-02",
            ),
        )

    def test_complete_window_starts(self) -> None:
        self.assertEqual(
            window_start_indices(
                0,
                1280,
                512,
                256,
            ),
            [
                0,
                256,
                512,
                768,
            ],
        )

    def test_short_record_returns_no_windows(self) -> None:
        self.assertEqual(
            window_start_indices(
                0,
                400,
                512,
                256,
            ),
            [],
        )

    def test_invalid_window_arguments(self) -> None:
        with self.assertRaises(
            ValueError
        ):
            window_start_indices(
                -1,
                1000,
                512,
                256,
            )

        with self.assertRaises(
            ValueError
        ):
            window_start_indices(
                0,
                1000,
                512,
                0,
            )

    def test_contiguous_false_runs(self) -> None:
        mask = np.array(
            [
                True,
                False,
                False,
                True,
                False,
                True,
            ],
            dtype=bool,
        )

        self.assertEqual(
            contiguous_false_runs(
                mask
            ),
            [
                (
                    1,
                    3,
                ),
                (
                    4,
                    5,
                ),
            ],
        )

    def test_short_internal_gap_is_interpolated(self) -> None:
        values = np.array(
            [
                1.0,
                np.nan,
                np.nan,
                4.0,
            ]
        )

        valid = np.isfinite(
            values
        )

        cleaned = interpolate_short_invalid_runs(
            values,
            valid,
            maximum_gap_samples=2,
        )

        np.testing.assert_allclose(
            cleaned,
            np.array(
                [
                    1.0,
                    2.0,
                    3.0,
                    4.0,
                ]
            ),
        )

    def test_long_gap_remains_invalid(self) -> None:
        values = np.array(
            [
                1.0,
                np.nan,
                np.nan,
                np.nan,
                5.0,
            ]
        )

        valid = np.isfinite(
            values
        )

        cleaned = interpolate_short_invalid_runs(
            values,
            valid,
            maximum_gap_samples=2,
        )

        self.assertTrue(
            np.isnan(
                cleaned[
                    1:4
                ]
            ).all()
        )

    def test_edge_gap_remains_invalid(self) -> None:
        values = np.array(
            [
                np.nan,
                2.0,
                3.0,
            ]
        )

        valid = np.isfinite(
            values
        )

        cleaned = interpolate_short_invalid_runs(
            values,
            valid,
            maximum_gap_samples=2,
        )

        self.assertTrue(
            np.isnan(
                cleaned[
                    0
                ]
            )
        )

    def test_nonpositive_pupil_values_are_invalid(self) -> None:
        cleaned, valid = preprocess_pupil(
            np.array(
                [
                    100.0,
                    0.0,
                    -5.0,
                    120.0,
                ]
            ),
            minimum_valid_value_exclusive=0.0,
            maximum_gap_samples=2,
        )

        self.assertEqual(
            valid.tolist(),
            [
                True,
                False,
                False,
                True,
            ],
        )

        np.testing.assert_allclose(
            cleaned,
            np.array(
                [
                    100.0,
                    106.66666666666667,
                    113.33333333333333,
                    120.0,
                ]
            ),
        )

    def test_configuration_contract(self) -> None:
        config = load_bbbd_config()

        self.assertEqual(
            config[
                "signals"
            ][
                "sampling_frequency_hz"
            ],
            128.0,
        )

        self.assertEqual(
            config[
                "windowing"
            ][
                "duration_seconds"
            ],
            4.0,
        )

        self.assertEqual(
            config[
                "windowing"
            ][
                "overlap_fraction"
            ],
            0.5,
        )

        self.assertEqual(
            len(
                config[
                    "signals"
                ][
                    "eeg"
                ][
                    "channels"
                ]
            ),
            8,
        )

        self.assertEqual(
            len(
                config[
                    "smoke_records"
                ]
            ),
            6,
        )


    def test_locked_direct_cohort(self) -> None:
        config = load_bbbd_config()

        participants = (
            load_direct_cohort_participants(
                config
            )
        )

        self.assertEqual(
            len(
                participants
            ),
            36,
        )

        self.assertIn(
            "experiment2::sub-01",
            participants,
        )

        self.assertIn(
            "experiment3::sub-02",
            participants,
        )

        self.assertNotIn(
            "experiment2::sub-08",
            participants,
        )

        self.assertNotIn(
            "experiment2::sub-29",
            participants,
        )



if __name__ == "__main__":
    unittest.main()