from __future__ import annotations

import json
import unittest
from pathlib import Path

import pandas as pd


JSON_PATH = Path(
    "_research_audit/"
    "bbbd_temporal_matched_08s_metrics_v1.json"
)

CSV_PATH = Path(
    "_research_audit/"
    "bbbd_temporal_matched_08s_metrics_v1.csv"
)


class BBBDTemporalMatchedEightSecondResultsAuditTests(
    unittest.TestCase
):

    @classmethod
    def setUpClass(cls) -> None:
        cls.audit = json.loads(
            JSON_PATH.read_text(
                encoding="utf-8"
            )
        )

        cls.metrics = pd.read_csv(
            CSV_PATH,
            low_memory=False,
        )

    def test_eight_second_numerical_audit_passed(self) -> None:
        self.assertEqual(
            self.audit["status"],
            "passed",
        )

        self.assertEqual(
            self.audit["numerical_mismatches"],
            0,
        )

        self.assertEqual(
            self.audit["duration_seconds"],
            8,
        )

    def test_all_28_groups_are_present(self) -> None:
        self.assertEqual(
            len(self.metrics),
            28,
        )

        self.assertEqual(
            set(
                self.metrics["path"]
            ),
            {
                "EEG",
                "ECG",
                "Pupil",
                "ECG_EEG",
                "ECG_Pupil",
                "EEG_Pupil",
                "ECG_EEG_Pupil",
            },
        )

        self.assertEqual(
            set(
                self.metrics[
                    "dataset_or_direction"
                ]
            ),
            {
                "experiment2",
                "experiment3",
                "experiment2_to_experiment3",
                "experiment3_to_experiment2",
            },
        )

    def test_duration_and_role_are_exact(self) -> None:
        self.assertEqual(
            set(
                self.metrics[
                    "duration_seconds"
                ].astype(int)
            ),
            {
                8,
            },
        )

        self.assertEqual(
            set(
                self.metrics[
                    "scientific_role"
                ]
            ),
            {
                "post_primary_temporal_sensitivity",
            },
        )

        self.assertEqual(
            self.audit[
                "scientific_role"
            ],
            "post-primary temporal sensitivity",
        )

    def test_metrics_and_intervals_are_bounded(self) -> None:
        metric_columns = [
            column
            for column in self.metrics.columns
            if any(
                token in column
                for token in [
                    "accuracy",
                    "macro_f1",
                    "roc_auc",
                    "pr_auc",
                ]
            )
            and not column.endswith(
                "_passed"
            )
        ]

        for column in metric_columns:
            values = pd.to_numeric(
                self.metrics[column],
                errors="raise",
            )

            self.assertTrue(
                values.notna().all()
            )

            self.assertTrue(
                (
                    values >= 0.0
                ).all()
            )

            self.assertTrue(
                (
                    values <= 1.0
                ).all()
            )

    def test_every_setting_has_all_paths_and_ranks(self) -> None:
        for _, frame in self.metrics.groupby(
            [
                "analysis_type",
                "dataset_or_direction",
            ]
        ):
            self.assertEqual(
                len(frame),
                7,
            )

            self.assertEqual(
                set(
                    frame[
                        "descriptive_rank_within_setting"
                    ].astype(int)
                ),
                set(
                    range(
                        1,
                        8,
                    )
                ),
            )

    def test_primary_four_second_analysis_is_preserved(self) -> None:
        self.assertFalse(
            self.audit[
                "primary_four_second_392_recording_analysis_replaced"
            ]
        )

        self.assertFalse(
            self.metrics[
                "primary_four_second_392_recording_analysis_replaced"
            ].astype(bool).any()
        )

    def test_complete_integrity_coverage(self) -> None:
        self.assertEqual(
            self.audit[
                "report_groups_checked"
            ],
            28,
        )

        self.assertEqual(
            self.audit[
                "bootstrap_rows_checked"
            ],
            140000,
        )

        self.assertEqual(
            self.audit[
                "report_files_checked"
            ],
            142,
        )

        for key in [
            "pooled_metrics_recomputed",
            "participant_metrics_recomputed",
            "participant_macro_metrics_recomputed",
            "threshold_decisions_recomputed",
            "confidence_intervals_recomputed",
            "manifest_hashes_verified",
            "registry_hashes_verified",
        ]:
            self.assertTrue(
                self.audit[key]
            )


if __name__ == "__main__":
    unittest.main()
