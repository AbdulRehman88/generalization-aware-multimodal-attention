from __future__ import annotations

import unittest

from src.features.ds003838_temporal_windows import (
    EXPECTED_WINDOW_NAMES,
    build_candidate_config,
    temporal_window_protocol,
)


class DS003838TemporalWindowTests(
    unittest.TestCase
):
    def setUp(self) -> None:
        self.config = {
            "datasets": {
                "ds003838": {
                    "window": {
                        "anchor":
                            "final_digit_offset",
                        "start_offset_seconds":
                            1.0,
                        "duration_seconds":
                            4.0,
                    }
                }
            },
            "training": {
                "ds003838_temporal_windows": {
                    "baseline": {
                        "anchor":
                            "first_digit_onset",
                        "start_offset_seconds":
                            -4.5,
                        "duration_seconds":
                            4.0,
                        "end_margin_seconds":
                            0.5,
                    },
                    "candidates": [
                        {
                            "name":
                                "late_encoding",
                            "anchor":
                                "final_digit_offset",
                            "start_offset_seconds":
                                -4.0,
                            "duration_seconds":
                                4.0,
                        },
                        {
                            "name":
                                "immediate_retention",
                            "anchor":
                                "final_digit_offset",
                            "start_offset_seconds":
                                0.0,
                            "duration_seconds":
                                4.0,
                        },
                        {
                            "name":
                                "delayed_retention",
                            "anchor":
                                "final_digit_offset",
                            "start_offset_seconds":
                                1.0,
                            "duration_seconds":
                                4.0,
                        },
                    ],
                }
            },
        }

    def test_protocol_preserves_candidate_order(self) -> None:
        protocol = temporal_window_protocol(
            self.config
        )

        self.assertEqual(
            [
                candidate["name"]
                for candidate
                in protocol["candidates"]
            ],
            EXPECTED_WINDOW_NAMES,
        )

    def test_candidate_configuration_is_isolated(self) -> None:
        protocol = temporal_window_protocol(
            self.config
        )

        candidate_config = build_candidate_config(
            self.config,
            protocol["candidates"][0],
        )

        candidate_window = candidate_config[
            "datasets"
        ][
            "ds003838"
        ][
            "window"
        ]

        self.assertEqual(
            candidate_window[
                "start_offset_seconds"
            ],
            -4.0,
        )

        self.assertEqual(
            self.config[
                "datasets"
            ][
                "ds003838"
            ][
                "window"
            ][
                "start_offset_seconds"
            ],
            1.0,
        )

    def test_modified_window_is_rejected(self) -> None:
        self.config[
            "training"
        ][
            "ds003838_temporal_windows"
        ][
            "candidates"
        ][0][
            "start_offset_seconds"
        ] = -3.0

        with self.assertRaises(ValueError):
            temporal_window_protocol(
                self.config
            )


if __name__ == "__main__":
    unittest.main()