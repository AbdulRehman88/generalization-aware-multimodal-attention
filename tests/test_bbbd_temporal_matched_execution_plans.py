from __future__ import annotations

import json
import unittest
from pathlib import Path


PLAN_PATHS = {
    4: Path(
        "_research_audit/"
        "bbbd_temporal_matched_execution_plan_04s_v1.json"
    ),
    8: Path(
        "_research_audit/"
        "bbbd_temporal_matched_execution_plan_08s_v1.json"
    ),
    16: Path(
        "_research_audit/"
        "bbbd_temporal_matched_execution_plan_16s_v1.json"
    ),
    24: Path(
        "_research_audit/"
        "bbbd_temporal_matched_execution_plan_24s_v1.json"
    ),
    32: Path(
        "_research_audit/"
        "bbbd_temporal_matched_execution_plan_32s_v1.json"
    ),
}


def canonical(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    )


class BBBDTemporalMatchedExecutionPlanTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.plans = {
            duration: json.loads(
                path.read_text(
                    encoding="utf-8"
                )
            )
            for duration, path in PLAN_PATHS.items()
        }

    def test_exact_duration_grid_and_cohort(self) -> None:
        self.assertEqual(
            sorted(self.plans),
            [4, 8, 16, 24, 32],
        )

        for duration, plan in self.plans.items():
            context = plan[
                "temporal_sensitivity_context"
            ]

            self.assertEqual(
                context["duration_seconds"],
                duration,
            )

            self.assertEqual(
                context["participants"],
                36,
            )

            self.assertEqual(
                context["recordings"],
                387,
            )

    def test_no_performance_or_fitting_preceded_plans(self) -> None:
        for plan in self.plans.values():
            self.assertFalse(
                plan["performance_observed"]
            )

            self.assertFalse(
                plan["feature_selector_fitted"]
            )

            self.assertFalse(
                plan["classifier_fitted"]
            )

    def test_exact_fit_totals(self) -> None:
        expected = {
            "candidate_model_fits": 1596,
            "cross_experiment_directions": 2,
            "selected_candidate_final_refits": 266,
            "shap_selector_fits": 266,
            "total_estimator_fits": 2128,
            "within_outer_folds": 36,
        }

        for plan in self.plans.values():
            self.assertEqual(
                plan["fit_totals"],
                expected,
            )

            self.assertEqual(
                len(
                    plan[
                        "within_experiment_folds"
                    ]
                ),
                36,
            )

            self.assertEqual(
                len(
                    plan[
                        "cross_experiment_directions"
                    ]
                ),
                2,
            )

    def test_participant_roles_are_identical(self) -> None:
        def roles(plan: dict) -> dict:
            return {
                "within": [
                    {
                        "dataset":
                            fold["dataset"],
                        "test":
                            fold[
                                "outer_test_participant"
                            ],
                        "training":
                            fold[
                                "inner_training_participants"
                            ],
                        "validation":
                            fold[
                                "inner_validation_participants"
                            ],
                    }
                    for fold in plan[
                        "within_experiment_folds"
                    ]
                ],
                "cross": [
                    {
                        "name":
                            direction["name"],
                        "training":
                            direction[
                                "source_training_participants"
                            ],
                        "validation":
                            direction[
                                "source_validation_participants"
                            ],
                    }
                    for direction in plan[
                        "cross_experiment_directions"
                    ]
                ],
            }

        reference = canonical(
            roles(
                self.plans[4]
            )
        )

        for duration in [
            8,
            16,
            24,
            32,
        ]:
            self.assertEqual(
                canonical(
                    roles(
                        self.plans[duration]
                    )
                ),
                reference,
            )

    def test_four_second_role_is_sensitivity_reference(self) -> None:
        self.assertEqual(
            self.plans[4][
                "temporal_sensitivity_context"
            ][
                "role"
            ],
            "matched_sensitivity_reference",
        )

        self.assertFalse(
            self.plans[4][
                "temporal_sensitivity_context"
            ][
                "primary_four_second_392_recording_result_replaced"
            ]
        )


if __name__ == "__main__":
    unittest.main()
