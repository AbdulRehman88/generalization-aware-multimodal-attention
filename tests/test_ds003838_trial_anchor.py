from __future__ import annotations

import unittest

from src.preprocessing.ds003838_revision import (
    trial_anchor_time,
)


class DS003838TrialAnchorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.trial = {
            "final_offset": 20.5,
            "events": [
                {
                    "time": 10.25,
                },
                {
                    "time": 12.0,
                },
            ],
        }

    def test_final_digit_offset_anchor(self) -> None:
        self.assertEqual(
            trial_anchor_time(
                self.trial,
                "final_digit_offset",
            ),
            20.5,
        )

    def test_first_digit_onset_anchor(self) -> None:
        self.assertEqual(
            trial_anchor_time(
                self.trial,
                "first_digit_onset",
            ),
            10.25,
        )

    def test_unknown_anchor_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            trial_anchor_time(
                self.trial,
                "unknown",
            )


if __name__ == "__main__":
    unittest.main()
