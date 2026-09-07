from __future__ import annotations

import unittest

import pandas as pd

from src.features.ds003838_temporal_window_cohort import (
    validate_window_alignment,
)


def make_table(
    *,
    segment_suffix: str = "",
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "segment_id":
                    f"p1_memory_001{segment_suffix}",
                "participant": "p1",
                "phase": "memory",
                "label": 0,
                "window_index": 0,
                "pupil_available": True,
                "delta_feature": 1.0,
            },
            {
                "segment_id":
                    f"p1_memory_002{segment_suffix}",
                "participant": "p1",
                "phase": "memory",
                "label": 1,
                "window_index": 1,
                "pupil_available": True,
                "delta_feature": 2.0,
            },
        ]
    )


class DS003838TemporalWindowCohortTests(
    unittest.TestCase
):
    def test_identical_metadata_alignment_passes(self) -> None:
        table = make_table()

        validate_window_alignment(
            {
                "late_encoding":
                    table.copy(),
                "immediate_retention":
                    table.copy(),
                "delayed_retention":
                    table.copy(),
            }
        )

    def test_segment_mismatch_is_rejected(self) -> None:
        with self.assertRaises(
            RuntimeError
        ):
            validate_window_alignment(
                {
                    "late_encoding":
                        make_table(),
                    "immediate_retention":
                        make_table(
                            segment_suffix="_wrong"
                        ),
                    "delayed_retention":
                        make_table(),
                }
            )

    def test_window_order_change_is_rejected(self) -> None:
        table = make_table()

        with self.assertRaises(
            RuntimeError
        ):
            validate_window_alignment(
                {
                    "immediate_retention":
                        table.copy(),
                    "late_encoding":
                        table.copy(),
                    "delayed_retention":
                        table.copy(),
                }
            )


if __name__ == "__main__":
    unittest.main()