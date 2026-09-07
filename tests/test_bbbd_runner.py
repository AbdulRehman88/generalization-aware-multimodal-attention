from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.bbbd_protocol import (
    load_bbbd_protocol_config,
)
from src.evaluation.bbbd_runner import (
    candidate_rank_key,
    evaluate_locked_split,
    feature_columns,
    recording_balanced_sample_indices,
    run_synthetic_smoke,
    select_participants,
    synthetic_table,
    training_only_feature_cleanup,
    validate_feature_frame,
)


class BBBDEvaluationRunnerTests(
    unittest.TestCase
):
    def test_training_cleanup_removes_constant_and_duplicate_features(
        self,
    ) -> None:
        frame = pd.DataFrame(
            {
                "a": [
                    1.0,
                    2.0,
                    3.0,
                ],
                "b": [
                    5.0,
                    5.0,
                    5.0,
                ],
                "c": [
                    1.0,
                    2.0,
                    3.0,
                ],
                "d": [
                    3.0,
                    2.0,
                    1.0,
                ],
            }
        )

        result = training_only_feature_cleanup(
            frame,
            [
                "a",
                "b",
                "c",
                "d",
            ],
        )

        self.assertEqual(
            result[
                "constant_features"
            ],
            [
                "b",
            ],
        )

        self.assertEqual(
            result[
                "duplicate_features"
            ],
            [
                {
                    "feature":
                        "c",
                    "duplicate_of":
                        "a",
                }
            ],
        )

        self.assertEqual(
            result[
                "retained_features"
            ],
            [
                "a",
                "d",
            ],
        )

    def test_recording_balanced_sample_is_deterministic(
        self,
    ) -> None:
        frame = pd.DataFrame(
            {
                "recording_id": (
                    [
                        "r1"
                    ]
                    * 10
                    + [
                        "r2"
                    ]
                    * 3
                    + [
                        "r3"
                    ]
                    * 8
                ),
                "segment_id": [
                    f"s{index:03d}"
                    for index in range(
                        21
                    )
                ],
            }
        )

        first = recording_balanced_sample_indices(
            frame,
            maximum_rows=
                9,
            random_seed=
                3407,
        )

        second = recording_balanced_sample_indices(
            frame,
            maximum_rows=
                9,
            random_seed=
                3407,
        )

        np.testing.assert_array_equal(
            first,
            second,
        )

        selected = frame.iloc[
            first
        ]

        self.assertEqual(
            set(
                selected[
                    "recording_id"
                ]
            ),
            {
                "r1",
                "r2",
                "r3",
            },
        )

    def test_candidate_ranking_contract(
        self,
    ) -> None:
        common = {
            "validation_balanced_accuracy":
                0.8,
            "validation_macro_f1":
                0.75,
            "effective_top_k":
                20,
            "parameters": {
                "x":
                    1,
            },
        }

        extra_trees = {
            **common,
            "model":
                "extra_trees",
            "candidate_id":
                "a",
        }

        xgboost = {
            **common,
            "model":
                "xgboost",
            "candidate_id":
                "b",
        }

        selected = sorted(
            [
                xgboost,
                extra_trees,
            ],
            key=
                candidate_rank_key,
        )[
            0
        ]

        self.assertEqual(
            selected[
                "model"
            ],
            "extra_trees",
        )

    def test_synthetic_table_contract(
        self,
    ) -> None:
        frame = synthetic_table(
            datasets=[
                "a",
                "b",
            ],
            participants_per_dataset=
                4,
            random_seed=
                3407,
            feature_count=
                6,
        )

        features = validate_feature_frame(
            frame,
            expected_feature_count=
                6,
        )

        self.assertEqual(
            len(
                features
            ),
            6,
        )

        self.assertEqual(
            frame[
                "participant"
            ].nunique(),
            8,
        )

        self.assertEqual(
            frame[
                "recording_id"
            ].nunique(),
            16,
        )

    def test_participant_selection_is_exact(
        self,
    ) -> None:
        frame = synthetic_table(
            datasets=[
                "a",
            ],
            participants_per_dataset=
                4,
            random_seed=
                3407,
            feature_count=
                4,
        )

        participants = sorted(
            frame[
                "participant"
            ].unique()
        )

        result = select_participants(
            frame,
            participants[
                :2
            ],
        )

        self.assertEqual(
            set(
                result[
                    "participant"
                ]
            ),
            set(
                participants[
                    :2
                ]
            ),
        )

    def test_incomplete_output_is_not_silently_overwritten(
        self,
    ) -> None:
        frame = synthetic_table(
            datasets=[
                "a",
            ],
            participants_per_dataset=
                6,
            random_seed=
                3407,
            feature_count=
                4,
        )

        participants = sorted(
            frame[
                "participant"
            ].unique()
        )

        config = load_bbbd_protocol_config()

        evaluation = copy.deepcopy(
            config[
                "evaluation"
            ]
        )

        evaluation[
            "training"
        ][
            "models"
        ][
            "extra_trees"
        ][
            "n_estimators"
        ] = [
            5,
        ]

        evaluation[
            "training"
        ][
            "models"
        ][
            "xgboost"
        ][
            "n_estimators"
        ] = [
            5,
        ]

        evaluation[
            "feature_selection"
        ][
            "top_k_values"
        ] = [
            "all",
        ]

        evaluation[
            "feature_selection"
        ][
            "maximum_shap_samples"
        ] = 20

        features = feature_columns(
            frame
        )

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(
                temporary
            ) / "split"

            output.mkdir(
                parents=True
            )

            (
                output
                / "partial.txt"
            ).write_text(
                "partial",
                encoding="utf-8",
            )

            with self.assertRaises(
                RuntimeError
            ):
                evaluate_locked_split(
                    training_frame=
                        select_participants(
                            frame,
                            participants[
                                :3
                            ],
                        ),
                    validation_frame=
                        select_participants(
                            frame,
                            participants[
                                3:5
                            ],
                        ),
                    test_frame=
                        select_participants(
                            frame,
                            participants[
                                5:
                            ],
                        ),
                    candidate_features=
                        features,
                    evaluation_config=
                        evaluation,
                    output_directory=
                        output,
                    split_identity={
                        "test":
                            "incomplete",
                    },
                    save_selected_model=
                        False,
                )

    def test_full_synthetic_runner_smoke(
        self,
    ) -> None:
        result = run_synthetic_smoke()

        self.assertTrue(
            result[
                "nested_split_completed"
            ]
        )

        self.assertTrue(
            result[
                "transfer_split_completed"
            ]
        )

        self.assertTrue(
            result[
                "resume_verified"
            ]
        )

        self.assertFalse(
            result[
                "bbbd_classifier_fitted"
            ]
        )

        self.assertFalse(
            result[
                "outer_test_performance_observed"
            ]
        )

        self.assertFalse(
            result[
                "target_experiment_performance_observed"
            ]
        )


if __name__ == "__main__":
    unittest.main()