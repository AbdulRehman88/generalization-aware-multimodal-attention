from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.evaluation.internal_grouped_pilot import (
    METADATA_COLUMNS,
)
from src.features.ds003838_trial_relative_features import (
    align_feature_tables,
    build_baseline_config,
    build_delta_table,
)
from src.features.internal_xr_revision_features import (
    feature_columns,
)


def sample_table(
    *,
    feature_offset: float,
) -> pd.DataFrame:
    rows = []

    for index in range(3):
        rows.append(
            {
                "segment_id":
                    f"p1_memory_{index + 1:03d}",
                "participant": "p1",
                "phase": "memory",
                "label": index,
                "window_index": index,
                "pupil_available": True,
                "ecg_mean":
                    feature_offset + index,
                "eeg_AF7_mean":
                    feature_offset + 2 * index,
            }
        )

    return pd.DataFrame(
        rows
    )


class DS003838TrialRelativeFeatureTests(
    unittest.TestCase
):
    def test_baseline_configuration_is_trial_local(self) -> None:
        config = {
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
            }
        }

        baseline = build_baseline_config(
            config
        )

        window = baseline[
            "datasets"
        ][
            "ds003838"
        ][
            "window"
        ]

        self.assertEqual(
            window["anchor"],
            "first_digit_onset",
        )

        self.assertEqual(
            window[
                "start_offset_seconds"
            ],
            -4.5,
        )

        self.assertEqual(
            window["duration_seconds"],
            4.0,
        )

        self.assertEqual(
            config[
                "datasets"
            ][
                "ds003838"
            ][
                "window"
            ][
                "anchor"
            ],
            "final_digit_offset",
        )

    def test_delta_table_is_post_minus_baseline(self) -> None:
        post = sample_table(
            feature_offset=10.0
        )

        baseline = sample_table(
            feature_offset=3.0
        )

        delta = build_delta_table(
            post,
            baseline,
        )

        features = feature_columns(
            delta
        )

        self.assertEqual(
            features,
            [
                "delta_ecg_mean",
                "delta_eeg_AF7_mean",
            ],
        )

        np.testing.assert_allclose(
            delta[
                features
            ].to_numpy(dtype=float),
            np.full(
                (3, 2),
                7.0,
            ),
        )

    def test_alignment_rejects_segment_mismatch(self) -> None:
        post = sample_table(
            feature_offset=10.0
        )

        baseline = sample_table(
            feature_offset=3.0
        )

        baseline.loc[
            0,
            "segment_id",
        ] = "wrong_segment"

        with self.assertRaises(
            RuntimeError
        ):
            align_feature_tables(
                post,
                baseline,
            )


if __name__ == "__main__":
    unittest.main()