from __future__ import annotations

import unittest

import numpy as np

from src.features.internal_xr_long_window_features import (
    build_long_window_plan,
    concatenate_signal_blocks,
    duration_directory_name,
)


class InternalXRLongWindowFeatureTests(
    unittest.TestCase
):
    def test_exact_16_second_half_overlap_plan(
        self,
    ) -> None:
        plan = build_long_window_plan(
            75,
            source_window_seconds=4,
            duration_seconds=16,
            overlap_fraction=0.50,
        )

        self.assertEqual(
            plan.source_blocks_per_window,
            4,
        )

        self.assertEqual(
            plan.source_blocks_per_stride,
            2,
        )

        self.assertEqual(
            plan.stride_seconds,
            8,
        )

        self.assertEqual(
            len(plan.start_indices),
            36,
        )

        self.assertEqual(
            plan.start_indices[:3],
            (
                0,
                2,
                4,
            ),
        )

        self.assertEqual(
            plan.start_indices[-1],
            70,
        )

    def test_exact_24_second_half_overlap_plan(
        self,
    ) -> None:
        plan = build_long_window_plan(
            105,
            source_window_seconds=4,
            duration_seconds=24,
            overlap_fraction=0.50,
        )

        self.assertEqual(
            plan.source_blocks_per_window,
            6,
        )

        self.assertEqual(
            plan.source_blocks_per_stride,
            3,
        )

        self.assertEqual(
            len(plan.start_indices),
            34,
        )

        self.assertEqual(
            plan.start_indices[-1],
            99,
        )

    def test_20_second_half_overlap_is_rejected_on_four_second_grid(
        self,
    ) -> None:
        with self.assertRaises(
            ValueError
        ):
            build_long_window_plan(
                105,
                source_window_seconds=4,
                duration_seconds=20,
                overlap_fraction=0.50,
            )

    def test_signal_block_concatenation_preserves_order(
        self,
    ) -> None:
        signal = np.arange(
            3
            * 4
            * 2,
            dtype=float,
        ).reshape(
            3,
            4,
            2,
        )

        combined = concatenate_signal_blocks(
            signal,
            start_index=1,
            block_count=2,
        )

        self.assertEqual(
            combined.shape,
            (
                8,
                2,
            ),
        )

        np.testing.assert_array_equal(
            combined,
            signal[
                1:3
            ].reshape(
                8,
                2,
            ),
        )

    def test_directory_name_is_stable(
        self,
    ) -> None:
        self.assertEqual(
            duration_directory_name(
                16,
                0.50,
            ),
            "window_16s_overlap_50pct",
        )


if __name__ == "__main__":
    unittest.main()