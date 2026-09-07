from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from src.evaluation.bbbd_reporting import (
    combine_split_recording_predictions,
    fold_aware_participant_cluster_bootstrap,
    fold_aware_participant_macro_metrics,
    fold_aware_recording_metrics,
    run_synthetic_smoke,
    synthetic_fold_aware_predictions,
    validate_fold_aware_recording_predictions,
    write_fold_aware_report,
)


class BBBDFoldAwareReportingTests(
    unittest.TestCase
):
    def test_fold_specific_thresholds_are_used(
        self,
    ) -> None:
        frame = synthetic_fold_aware_predictions()

        metrics = fold_aware_recording_metrics(
            frame
        )

        self.assertAlmostEqual(
            metrics[
                "balanced_accuracy"
            ],
            1.0,
        )

        global_predictions = (
            frame[
                "probability"
            ].to_numpy(
                dtype=float
            )
            >= 0.5
        ).astype(int)

        global_score = balanced_accuracy_score(
            frame[
                "label"
            ].astype(int),
            global_predictions,
        )

        self.assertLess(
            global_score,
            1.0,
        )

    def test_probability_metrics_are_not_threshold_shifted(
        self,
    ) -> None:
        frame = synthetic_fold_aware_predictions()

        original = frame[
            "probability"
        ].copy()

        metrics = fold_aware_recording_metrics(
            frame
        )

        pd.testing.assert_series_equal(
            frame[
                "probability"
            ],
            original,
        )

        self.assertAlmostEqual(
            metrics[
                "roc_auc"
            ],
            1.0,
        )

        self.assertAlmostEqual(
            metrics[
                "pr_auc"
            ],
            1.0,
        )

    def test_decision_threshold_mismatch_is_rejected(
        self,
    ) -> None:
        frame = synthetic_fold_aware_predictions()

        frame.loc[
            0,
            "predicted_label",
        ] = 1

        with self.assertRaises(
            ValueError
        ):
            validate_fold_aware_recording_predictions(
                frame
            )

    def test_participant_macro_metrics_are_fold_aware(
        self,
    ) -> None:
        frame = synthetic_fold_aware_predictions()

        metrics = fold_aware_participant_macro_metrics(
            frame
        )

        self.assertEqual(
            metrics[
                "participant_count"
            ],
            3.0,
        )

        self.assertAlmostEqual(
            metrics[
                "balanced_accuracy"
            ],
            1.0,
        )

        self.assertAlmostEqual(
            metrics[
                "macro_f1"
            ],
            1.0,
        )

    def test_participant_bootstrap_is_deterministic(
        self,
    ) -> None:
        frame = synthetic_fold_aware_predictions()

        first = fold_aware_participant_cluster_bootstrap(
            frame,
            repetitions=
                30,
            random_seed=
                3407,
        )

        second = fold_aware_participant_cluster_bootstrap(
            frame,
            repetitions=
                30,
            random_seed=
                3407,
        )

        pd.testing.assert_frame_equal(
            first,
            second,
        )

        self.assertEqual(
            len(
                first
            ),
            30,
        )

        self.assertTrue(
            np.isfinite(
                first.drop(
                    columns=[
                        "repetition",
                    ]
                ).to_numpy(
                    dtype=float
                )
            ).all()
        )

    def test_combining_completed_splits_preserves_thresholds(
        self,
    ) -> None:
        frame = synthetic_fold_aware_predictions()

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(
                temporary
            )

            directories = []

            for index, participant in enumerate(
                sorted(
                    frame[
                        "participant"
                    ].unique()
                )
            ):
                directory = (
                    root
                    / f"split-{index:02d}"
                )

                directory.mkdir(
                    parents=True
                )

                participant_frame = frame.loc[
                    frame[
                        "participant"
                    ]
                    == participant
                ].copy()

                participant_frame.to_csv(
                    directory
                    / "recording_predictions.csv",
                    index=False,
                )

                completion = {
                    "completed":
                        True,
                    "selected_threshold":
                        float(
                            participant_frame[
                                "threshold"
                            ].iloc[
                                0
                            ]
                        ),
                    "split_sha256":
                        f"split-{index:02d}",
                }

                (
                    directory
                    / "complete.json"
                ).write_text(
                    json.dumps(
                        completion
                    ),
                    encoding="utf-8",
                )

                directories.append(
                    directory
                )

            combined = combine_split_recording_predictions(
                directories
            )

        self.assertEqual(
            len(
                combined
            ),
            6,
        )

        self.assertEqual(
            set(
                combined[
                    "threshold"
                ]
            ),
            {
                0.3,
                0.5,
                0.7,
            },
        )

    def test_report_output_contract(
        self,
    ) -> None:
        frame = synthetic_fold_aware_predictions()

        with tempfile.TemporaryDirectory() as temporary:
            output = (
                Path(
                    temporary
                )
                / "report"
            )

            summary = write_fold_aware_report(
                recording_predictions=
                    frame,
                output_directory=
                    output,
                protocol_identity=
                    "synthetic",
                bootstrap_repetitions=
                    25,
                bootstrap_seed=
                    3407,
                confidence_level=
                    0.95,
            )

            expected_files = {
                "confidence_intervals.csv",
                "participant_bootstrap.csv",
                "participant_metrics.csv",
                "recording_predictions.csv",
                "summary.json",
            }

            observed_files = {
                path.name
                for path in output.iterdir()
                if path.is_file()
            }

            self.assertEqual(
                observed_files,
                expected_files,
            )

            self.assertAlmostEqual(
                summary[
                    "pooled_recording_metrics"
                ][
                    "balanced_accuracy"
                ],
                1.0,
            )

    def test_complete_synthetic_reporting_smoke(
        self,
    ) -> None:
        result = run_synthetic_smoke()

        self.assertFalse(
            result[
                "bbbd_performance_observed"
            ]
        )

        self.assertTrue(
            result[
                "participant_bootstrap_deterministic"
            ]
        )

        self.assertFalse(
            result[
                "probabilities_shifted"
            ]
        )

        self.assertLess(
            result[
                "incorrect_global_threshold_balanced_accuracy"
            ],
            result[
                "fold_aware_balanced_accuracy"
            ],
        )


if __name__ == "__main__":
    unittest.main()