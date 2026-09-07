from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.features.ds003838_revision_features import (
    empty_feature_table,
    fuse_feature_tables,
    parse_boolean,
)
from src.features.internal_xr_revision_features import (
    METADATA_COLUMNS,
)


class DS003838RevisionFeatureTests(unittest.TestCase):
    def test_boolean_parser(self) -> None:
        self.assertTrue(
            parse_boolean("True")
        )

        self.assertFalse(
            parse_boolean("False")
        )

        with self.assertRaises(ValueError):
            parse_boolean("unknown")

    def test_empty_feature_table_preserves_schema(self) -> None:
        table = empty_feature_table(
            [
                "pupil_feature_a",
                "pupil_feature_b",
            ]
        )

        self.assertEqual(
            len(table),
            0,
        )

        self.assertEqual(
            table.columns.tolist(),
            [
                *METADATA_COLUMNS,
                "pupil_feature_a",
                "pupil_feature_b",
            ],
        )

    def test_empty_pupil_subset_can_be_fused(self) -> None:
        pupil = empty_feature_table(
            ["pupil_feature"]
        )

        eeg = pd.DataFrame(
            {
                "segment_id": ["a", "b"],
                "participant": ["p1", "p1"],
                "phase": ["memory", "memory"],
                "label": [0, 1],
                "window_index": [0, 1],
                "pupil_available": [False, False],
                "eeg_feature": [1.0, 2.0],
            }
        )

        fused = fuse_feature_tables(
            pupil,
            [eeg],
            "empty_pupil_fusion",
        )

        self.assertEqual(
            len(fused),
            0,
        )

        self.assertEqual(
            fused.columns.tolist(),
            [
                *METADATA_COLUMNS,
                "pupil_feature",
                "eeg_feature",
            ],
        )

    def test_fusion_preserves_base_subset_order(self) -> None:
        full = pd.DataFrame(
            {
                "segment_id": ["a", "b", "c"],
                "participant": ["p1", "p1", "p1"],
                "phase": ["memory"] * 3,
                "label": [0, 1, 2],
                "window_index": [0, 1, 2],
                "pupil_available": [False, True, True],
                "eeg_feature": [1.0, 2.0, 3.0],
            }
        )

        subset = pd.DataFrame(
            {
                "segment_id": ["c", "b"],
                "participant": ["p1", "p1"],
                "phase": ["memory", "memory"],
                "label": [2, 1],
                "window_index": [2, 1],
                "pupil_available": [True, True],
                "pupil_feature": [30.0, 20.0],
            }
        )

        fused = fuse_feature_tables(
            subset,
            [full],
            "synthetic",
        )

        self.assertEqual(
            fused["segment_id"].tolist(),
            ["c", "b"],
        )

        self.assertEqual(
            fused["eeg_feature"].tolist(),
            [3.0, 2.0],
        )

        self.assertEqual(
            fused["pupil_feature"].tolist(),
            [30.0, 20.0],
        )

        self.assertEqual(
            fused.columns[
                :len(METADATA_COLUMNS)
            ].tolist(),
            METADATA_COLUMNS,
        )

        feature_values = fused[
            ["eeg_feature", "pupil_feature"]
        ].to_numpy(dtype=float)

        self.assertTrue(
            np.isfinite(feature_values).all()
        )


if __name__ == "__main__":
    unittest.main()