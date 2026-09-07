from __future__ import annotations

import json
import unittest
from pathlib import Path

import pandas as pd


JSON_PATH = Path(
    "_research_audit/"
    "bbbd_temporal_matched_all_durations_v1.json"
)

CSV_PATH = Path(
    "_research_audit/"
    "bbbd_temporal_matched_all_durations_v1.csv"
)


class BBBDTemporalMatchedAllDurationsAuditTests(
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

    def test_aggregate_audit_passed(self) -> None:
        self.assertEqual(
            self.audit["status"],
            "passed",
        )

        self.assertEqual(
            self.audit["numerical_mismatches"],
            0,
        )

    def test_exact_duration_grid(self) -> None:
        self.assertEqual(
            self.audit["durations_seconds"],
            [4, 8, 16, 24, 32],
        )

        self.assertEqual(
            set(
                self.metrics[
                    "duration_seconds"
                ].astype(int)
            ),
            {4, 8, 16, 24, 32},
        )

    def test_exact_row_counts(self) -> None:
        self.assertEqual(
            len(self.metrics),
            140,
        )

        duration_counts = (
            self.metrics[
                "duration_seconds"
            ]
            .astype(int)
            .value_counts()
            .to_dict()
        )

        self.assertEqual(
            duration_counts,
            {
                4: 28,
                8: 28,
                16: 28,
                24: 28,
                32: 28,
            },
        )

    def test_exact_path_and_setting_grid(self) -> None:
        self.assertEqual(
            set(self.metrics["path"]),
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

        settings = set(
            zip(
                self.metrics[
                    "analysis_type"
                ],
                self.metrics[
                    "dataset_or_direction"
                ],
            )
        )

        self.assertEqual(
            settings,
            {
                (
                    "within_experiment",
                    "experiment2",
                ),
                (
                    "within_experiment",
                    "experiment3",
                ),
                (
                    "cross_experiment",
                    "experiment2_to_experiment3",
                ),
                (
                    "cross_experiment",
                    "experiment3_to_experiment2",
                ),
            },
        )

    def test_four_second_reference_deltas_are_zero(self) -> None:
        reference = self.metrics.loc[
            self.metrics[
                "duration_seconds"
            ].astype(int)
            == 4
        ]

        for column in [
            "delta_balanced_accuracy_vs_04s",
            "delta_macro_f1_vs_04s",
            "delta_roc_auc_vs_04s",
            "delta_pr_auc_vs_04s",
        ]:
            self.assertTrue(
                (
                    reference[column].abs()
                    <= 1e-12
                ).all()
            )

    def test_all_setting_duration_ranks_are_complete(self) -> None:
        for _, frame in self.metrics.groupby(
            [
                "analysis_type",
                "dataset_or_direction",
                "duration_seconds",
            ]
        ):
            self.assertEqual(
                len(frame),
                7,
            )

            self.assertEqual(
                set(
                    frame[
                        "temporal_descriptive_rank_within_setting_duration"
                    ].astype(int)
                ),
                set(range(1, 8)),
            )

    def test_summary_record_counts(self) -> None:
        self.assertEqual(
            len(
                self.audit[
                    "best_path_by_setting_and_duration"
                ]
            ),
            20,
        )

        self.assertEqual(
            len(
                self.audit[
                    "descriptive_best_overall_by_setting"
                ]
            ),
            4,
        )

        self.assertEqual(
            len(
                self.audit[
                    "descriptive_best_duration_by_setting_path"
                ]
            ),
            28,
        )

    def test_no_new_fitting_or_inferential_claim(self) -> None:
        self.assertFalse(
            self.audit[
                "model_fitting_performed"
            ]
        )

        self.assertFalse(
            self.audit[
                "feature_selection_performed"
            ]
        )

        self.assertFalse(
            self.audit[
                "threshold_selection_performed"
            ]
        )

        self.assertFalse(
            self.audit[
                "inferential_duration_superiority_claimed"
            ]
        )

        self.assertFalse(
            self.audit[
                "minimum_reliable_duration_claimed"
            ]
        )

    def test_primary_four_second_analysis_is_preserved(self) -> None:
        self.assertFalse(
            self.audit[
                "primary_four_second_392_recording_analysis_replaced"
            ]
        )

    def test_source_evidence_is_complete(self) -> None:
        self.assertEqual(
            len(
                self.audit[
                    "source_evidence"
                ]
            ),
            5,
        )

        for source in self.audit[
            "source_evidence"
        ]:
            self.assertEqual(
                source[
                    "report_groups"
                ],
                28,
            )

            self.assertEqual(
                source[
                    "bootstrap_rows"
                ],
                140000,
            )

            self.assertEqual(
                len(
                    source[
                        "metrics_csv_sha256"
                    ]
                ),
                64,
            )

            self.assertEqual(
                len(
                    source[
                        "metrics_json_sha256"
                    ]
                ),
                64,
            )


if __name__ == "__main__":
    unittest.main()
