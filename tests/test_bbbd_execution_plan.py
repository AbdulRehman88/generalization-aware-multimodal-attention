from __future__ import annotations

import unittest

from src.evaluation.bbbd_execution_plan import (
    effective_top_k_candidates,
    enumerate_candidate_specs,
    expand_parameter_grid,
)
from src.evaluation.bbbd_protocol import (
    deterministic_source_split,
    load_bbbd_protocol_config,
)


class BBBDExecutionPlanTests(
    unittest.TestCase
):
    def test_source_split_is_deterministic_and_disjoint(
        self,
    ) -> None:
        participants = [
            f"experiment2::sub-{index:02d}"
            for index in range(
                1,
                21,
            )
        ]

        first = deterministic_source_split(
            participants,
            random_seed=
                3407,
            validation_fraction=
                0.20,
            minimum_validation_participants=
                2,
        )

        second = deterministic_source_split(
            participants,
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

        self.assertEqual(
            len(
                validation
            ),
            4,
        )

        self.assertFalse(
            set(
                training
            )
            & set(
                validation
            )
        )

        self.assertEqual(
            set(
                training
            )
            | set(
                validation
            ),
            set(
                participants
            ),
        )

    def test_effective_top_k_deduplicates_low_dimension_path(
        self,
    ) -> None:
        candidates = effective_top_k_candidates(
            16,
            [
                20,
                50,
                100,
                "all",
            ],
        )

        self.assertEqual(
            candidates,
            [
                {
                    "configured_top_k":
                        "all",
                    "effective_top_k":
                        16,
                }
            ],
        )

    def test_effective_top_k_retains_meaningful_pupil_options(
        self,
    ) -> None:
        candidates = effective_top_k_candidates(
            23,
            [
                20,
                50,
                100,
                "all",
            ],
        )

        self.assertEqual(
            [
                candidate[
                    "effective_top_k"
                ]
                for candidate in candidates
            ],
            [
                20,
                23,
            ],
        )

    def test_parameter_grid_expansion(
        self,
    ) -> None:
        result = expand_parameter_grid(
            {
                "depth": [
                    3,
                    6,
                ],
                "rate": [
                    0.05,
                ],
                "n_jobs":
                    -1,
            }
        )

        self.assertEqual(
            len(
                result
            ),
            2,
        )

        self.assertEqual(
            {
                candidate[
                    "depth"
                ]
                for candidate in result
            },
            {
                3,
                6,
            },
        )

    def test_locked_model_and_top_k_candidate_counts(
        self,
    ) -> None:
        config = load_bbbd_protocol_config()

        evaluation = config[
            "evaluation"
        ]

        expected = {
            "ECG": 2,
            "Pupil": 4,
            "ECG_Pupil": 4,
            "EEG": 8,
            "ECG_EEG": 8,
            "EEG_Pupil": 8,
            "ECG_EEG_Pupil": 8,
        }

        feature_counts = {
            "ECG": 16,
            "Pupil": 23,
            "ECG_Pupil": 39,
            "EEG": 232,
            "ECG_EEG": 248,
            "EEG_Pupil": 255,
            "ECG_EEG_Pupil": 271,
        }

        observed = {
            path:
                len(
                    enumerate_candidate_specs(
                        evaluation,
                        feature_count=
                            feature_count,
                    )
                )
            for path, feature_count
            in feature_counts.items()
        }

        self.assertEqual(
            observed,
            expected,
        )

        self.assertEqual(
            sum(
                observed.values()
            ),
            42,
        )

    def test_execution_contract_is_frozen_before_performance(
        self,
    ) -> None:
        config = load_bbbd_protocol_config()

        execution = config[
            "evaluation"
        ][
            "execution"
        ]

        stop_rule = config[
            "evaluation"
        ][
            "stop_rule"
        ]

        self.assertEqual(
            execution[
                "plan_version"
            ],
            1,
        )

        self.assertTrue(
            execution[
                "deduplicate_effective_top_k"
            ]
        )

        self.assertFalse(
            stop_rule[
                "execution_plan_may_change_after_performance"
            ]
        )

        self.assertFalse(
            config[
                "evaluation"
            ][
                "cross_experiment"
            ][
                "target_experiment_tuning"
            ]
        )


if __name__ == "__main__":
    unittest.main()