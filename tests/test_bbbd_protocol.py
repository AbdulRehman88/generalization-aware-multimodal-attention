from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.evaluation.bbbd_protocol import (
    aggregate_recording_probabilities,
    binary_probability_metrics,
    bootstrap_confidence_intervals,
    deterministic_inner_split,
    load_bbbd_protocol_config,
    participant_cluster_bootstrap,
    participant_macro_metrics,
    recording_equal_window_weights,
    select_recording_threshold,
    validate_role_disjointness,
)


class BBBDEvaluationProtocolTests(
    unittest.TestCase
):
    def test_deterministic_inner_split_is_disjoint_and_complete(
        self,
    ) -> None:
        participants = [
            f"experiment2::sub-{index:02d}"
            for index in range(
                1,
                21,
            )
        ]

        first = deterministic_inner_split(
            participants,
            outer_test_participant=
                "experiment2::sub-01",
            random_seed=
                3407,
            validation_fraction=
                0.20,
            minimum_validation_participants=
                2,
        )

        second = deterministic_inner_split(
            participants,
            outer_test_participant=
                "experiment2::sub-01",
            random_seed=
                3407,
            validation_fraction=
                0.20,
            minimum_validation_participants=
                2,
        )

        self.assertEqual(
            first,
            second,
        )

        training, validation = first

        self.assertFalse(
            set(
                training
            )
            & set(
                validation
            )
        )

        self.assertNotIn(
            "experiment2::sub-01",
            training,
        )

        self.assertNotIn(
            "experiment2::sub-01",
            validation,
        )

        self.assertEqual(
            set(
                training
            )
            | set(
                validation
            )
            | {
                "experiment2::sub-01"
            },
            set(
                participants
            ),
        )

        self.assertEqual(
            len(
                validation
            ),
            4,
        )

    def test_role_disjointness_accepts_contained_roles(
        self,
    ) -> None:
        training = pd.DataFrame(
            {
                "participant": [
                    "p1",
                    "p1",
                ],
                "recording_id": [
                    "r1",
                    "r1",
                ],
                "segment_id": [
                    "s1",
                    "s2",
                ],
            }
        )

        validation = pd.DataFrame(
            {
                "participant": [
                    "p2",
                ],
                "recording_id": [
                    "r2",
                ],
                "segment_id": [
                    "s3",
                ],
            }
        )

        test = pd.DataFrame(
            {
                "participant": [
                    "p3",
                ],
                "recording_id": [
                    "r3",
                ],
                "segment_id": [
                    "s4",
                ],
            }
        )

        validate_role_disjointness(
            {
                "training":
                    training,
                "validation":
                    validation,
                "test":
                    test,
            }
        )

    def test_role_disjointness_rejects_participant_overlap(
        self,
    ) -> None:
        training = pd.DataFrame(
            {
                "participant": [
                    "p1",
                ],
                "recording_id": [
                    "r1",
                ],
                "segment_id": [
                    "s1",
                ],
            }
        )

        test = pd.DataFrame(
            {
                "participant": [
                    "p1",
                ],
                "recording_id": [
                    "r2",
                ],
                "segment_id": [
                    "s2",
                ],
            }
        )

        with self.assertRaises(
            ValueError
        ):
            validate_role_disjointness(
                {
                    "training":
                        training,
                    "test":
                        test,
                }
            )

    def test_recording_equal_weights_sum_to_one(
        self,
    ) -> None:
        frame = pd.DataFrame(
            {
                "recording_id": [
                    "r1",
                    "r1",
                    "r2",
                    "r2",
                    "r2",
                    "r2",
                ]
            }
        )

        weights = recording_equal_window_weights(
            frame
        )

        result = pd.DataFrame(
            {
                "recording_id":
                    frame[
                        "recording_id"
                    ],
                "weight":
                    weights,
            }
        ).groupby(
            "recording_id"
        )[
            "weight"
        ].sum()

        np.testing.assert_allclose(
            result.to_numpy(
                dtype=float
            ),
            np.array(
                [
                    1.0,
                    1.0,
                ]
            ),
        )

        self.assertGreater(
            weights[
                0
            ],
            weights[
                2
            ],
        )

    def test_recording_probability_aggregation(
        self,
    ) -> None:
        frame = pd.DataFrame(
            {
                "dataset": [
                    "experiment2",
                    "experiment2",
                    "experiment2",
                    "experiment2",
                ],
                "participant": [
                    "p1",
                    "p1",
                    "p1",
                    "p1",
                ],
                "recording_id": [
                    "r1",
                    "r1",
                    "r2",
                    "r2",
                ],
                "task": [
                    "t1",
                    "t1",
                    "t2",
                    "t2",
                ],
                "label": [
                    0,
                    0,
                    1,
                    1,
                ],
                "probability": [
                    0.1,
                    0.3,
                    0.7,
                    0.9,
                ],
            }
        )

        result = aggregate_recording_probabilities(
            frame,
            threshold=
                0.5,
        )

        self.assertEqual(
            result[
                "recording_id"
            ].tolist(),
            [
                "r1",
                "r2",
            ],
        )

        np.testing.assert_allclose(
            result[
                "probability"
            ].to_numpy(
                dtype=float
            ),
            np.array(
                [
                    0.2,
                    0.8,
                ]
            ),
        )

        self.assertEqual(
            result[
                "predicted_label"
            ].tolist(),
            [
                0,
                1,
            ],
        )

        self.assertEqual(
            result[
                "window_count"
            ].tolist(),
            [
                2,
                2,
            ],
        )

    def test_binary_metrics_are_perfect(
        self,
    ) -> None:
        metrics = binary_probability_metrics(
            [
                0,
                0,
                1,
                1,
            ],
            [
                0.1,
                0.2,
                0.8,
                0.9,
            ],
            threshold=
                0.5,
        )

        for metric in [
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "roc_auc",
            "pr_auc",
        ]:
            self.assertAlmostEqual(
                metrics[
                    metric
                ],
                1.0,
            )

    def test_threshold_selection_uses_prespecified_tie_break(
        self,
    ) -> None:
        validation = pd.DataFrame(
            {
                "label": [
                    0,
                    0,
                    1,
                    1,
                ],
                "probability": [
                    0.1,
                    0.2,
                    0.8,
                    0.9,
                ],
            }
        )

        selected, table = select_recording_threshold(
            validation,
            candidates=[
                0.3,
                0.4,
                0.5,
                0.6,
                0.7,
            ],
        )

        self.assertEqual(
            selected,
            0.5,
        )

        self.assertEqual(
            table[
                "threshold"
            ].tolist(),
            [
                0.3,
                0.4,
                0.5,
                0.6,
                0.7,
            ],
        )

    def test_participant_macro_metrics(
        self,
    ) -> None:
        predictions = pd.DataFrame(
            {
                "participant": [
                    "p1",
                    "p1",
                    "p2",
                    "p2",
                ],
                "label": [
                    0,
                    1,
                    0,
                    1,
                ],
                "probability": [
                    0.1,
                    0.9,
                    0.2,
                    0.8,
                ],
            }
        )

        result = participant_macro_metrics(
            predictions,
            threshold=
                0.5,
        )

        self.assertEqual(
            result[
                "participant_count"
            ],
            2.0,
        )

        self.assertAlmostEqual(
            result[
                "balanced_accuracy"
            ],
            1.0,
        )

        self.assertAlmostEqual(
            result[
                "macro_f1"
            ],
            1.0,
        )

    def test_participant_bootstrap_is_deterministic(
        self,
    ) -> None:
        predictions = pd.DataFrame(
            {
                "participant": [
                    "p1",
                    "p1",
                    "p2",
                    "p2",
                    "p3",
                    "p3",
                ],
                "label": [
                    0,
                    1,
                    0,
                    1,
                    0,
                    1,
                ],
                "probability": [
                    0.1,
                    0.9,
                    0.2,
                    0.8,
                    0.3,
                    0.7,
                ],
            }
        )

        first = participant_cluster_bootstrap(
            predictions,
            repetitions=
                25,
            random_seed=
                3407,
            threshold=
                0.5,
        )

        second = participant_cluster_bootstrap(
            predictions,
            repetitions=
                25,
            random_seed=
                3407,
            threshold=
                0.5,
        )

        pd.testing.assert_frame_equal(
            first,
            second,
        )

        self.assertEqual(
            len(
                first
            ),
            25,
        )

        confidence = bootstrap_confidence_intervals(
            first,
            confidence_level=
                0.95,
        )

        self.assertFalse(
            confidence.empty
        )

    def test_configuration_contract(
        self,
    ) -> None:
        config = load_bbbd_protocol_config()

        evaluation = config[
            "evaluation"
        ]

        self.assertEqual(
            evaluation[
                "protocol_name"
            ],
            "bbbd_nested_participant_disjoint_v1",
        )

        self.assertEqual(
            evaluation[
                "within_experiment"
            ][
                "protocol"
            ],
            "nested_leave_one_participant_out",
        )

        self.assertFalse(
            evaluation[
                "within_experiment"
            ][
                "pooled_experiment_loso_primary"
            ]
        )

        self.assertFalse(
            evaluation[
                "cross_experiment"
            ][
                "target_experiment_tuning"
            ]
        )

        self.assertEqual(
            evaluation[
                "training"
            ][
                "window_weighting"
            ],
            "inverse_accepted_window_count_per_recording",
        )

        self.assertEqual(
            evaluation[
                "aggregation"
            ][
                "primary_decision_unit"
            ],
            "recording",
        )

        self.assertEqual(
            evaluation[
                "bootstrap"
            ][
                "unit"
            ],
            "participant",
        )

        self.assertEqual(
            evaluation[
                "bootstrap"
            ][
                "repetitions"
            ],
            5000,
        )

        self.assertTrue(
            evaluation[
                "stop_rule"
            ][
                "performance_chasing_prohibited"
            ]
        )


if __name__ == "__main__":
    unittest.main()