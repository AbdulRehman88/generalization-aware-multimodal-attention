from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.features.bbbd_temporal_matched_cohort import (
    EXPECTED_DURATIONS,
    boolean_mask,
    duration_directory_name,
    load_selected_registry,
)


class BBBDTemporalMatchedCohortTests(unittest.TestCase):

    def test_boolean_mask_is_strict(self) -> None:
        observed = boolean_mask(
            pd.Series(
                ["True", "false", "1", "0"]
            )
        )

        self.assertEqual(
            observed.tolist(),
            [True, False, True, False],
        )

        with self.assertRaises(ValueError):
            boolean_mask(
                pd.Series(["yes"])
            )

    def test_duration_directories_are_exact(self) -> None:
        self.assertEqual(
            [
                duration_directory_name(duration)
                for duration in EXPECTED_DURATIONS
            ],
            [
                "window_04s_matched_387",
                "window_08s_matched_387",
                "window_16s_matched_387",
                "window_24s_matched_387",
                "window_32s_matched_387",
            ],
        )

    def test_unknown_duration_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            duration_directory_name(12)

    def test_locked_registry_selects_exact_cohort(self) -> None:
        selected = load_selected_registry(
            (
                "artifacts/revision/manifests/"
                "bbbd_temporal_sensitivity_recording_registry_v1.csv"
            ),
            "temporal_common_complete_case",
        )

        self.assertEqual(
            len(selected),
            387,
        )

        self.assertEqual(
            selected["participant"].nunique(),
            36,
        )

        self.assertEqual(
            set(selected["dataset"]),
            {"experiment2", "experiment3"},
        )

        self.assertEqual(
            set(selected["label"].astype(int)),
            {0, 1},
        )

    def test_registry_duplicate_ids_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "registry.csv"

            pd.DataFrame(
                {
                    "recording_id": ["r1", "r1"],
                    "dataset": ["experiment2", "experiment2"],
                    "participant": ["experiment2::p1", "experiment2::p1"],
                    "label": [0, 1],
                    "selected": [True, True],
                }
            ).to_csv(
                path,
                index=False,
            )

            with self.assertRaises(RuntimeError):
                load_selected_registry(
                    path,
                    "selected",
                )


if __name__ == "__main__":
    unittest.main()
