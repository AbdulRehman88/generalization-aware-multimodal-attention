
import json
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

REGISTRY = (
    ROOT
    / "artifacts/revision/manifests/"
      "deep_learning_execution_registry_v2.csv"
)

PLAN = (
    ROOT
    / "_research_audit/deep_learning_execution_plan_v2.json"
)

DRY = (
    ROOT
    / "_research_audit/deep_learning_real_data_dry_run_v1.json"
)


class TestDeepLearningExecutionPlan(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        if not REGISTRY.is_file():
            raise RuntimeError(
                "Execution registry has not been generated."
            )

        if not PLAN.is_file():
            raise RuntimeError(
                "Execution-plan audit has not been generated."
            )

        if not DRY.is_file():
            raise RuntimeError(
                "Real-data dry-run audit has not been generated."
            )

        cls.registry = pd.read_csv(
            REGISTRY,
            low_memory=False,
        )

        cls.plan = json.loads(
            PLAN.read_text(
                encoding="utf-8"
            )
        )

        cls.dry = json.loads(
            DRY.read_text(
                encoding="utf-8"
            )
        )

    def test_registry_size(self):
        self.assertEqual(
            len(
                self.registry
            ),
            3000,
        )

    def test_stage_counts(self):
        counts = (
            self.registry[
                "stage"
            ]
            .value_counts()
            .to_dict()
        )

        self.assertEqual(
            counts,
            {
                "internal_conventional":
                    840,

                "internal_nested_loso":
                    312,

                "internal_subject_adaptive":
                    936,

                "bbbd_within_experiment":
                    864,

                "bbbd_cross_experiment":
                    48,
            },
        )

    def test_seeds(self):
        self.assertEqual(
            sorted(
                self.registry[
                    "seed"
                ].unique().tolist()
            ),
            [
                42,
                2026,
                3407,
            ],
        )

    def test_model_path_count(self):
        pairs = (
            self.registry[
                [
                    "model",
                    "path",
                ]
            ]
            .drop_duplicates()
        )

        self.assertEqual(
            len(
                pairs
            ),
            8,
        )

        shallow = pairs.loc[
            pairs[
                "model"
            ]
            == "ShallowConvNet"
        ]

        self.assertEqual(
            shallow[
                "path"
            ].tolist(),
            [
                "EEG"
            ],
        )

    def test_nested_candidate_durations(self):
        nested = self.registry.loc[
            self.registry[
                "stage"
            ]
            == "internal_nested_loso"
        ]

        self.assertTrue(
            (
                nested[
                    "candidate_durations_seconds"
                ]
                == "8;16;24;32"
            ).all()
        )

    def test_adaptive_depends_on_nested_loso(self):
        adaptive = self.registry.loc[
            self.registry[
                "stage"
            ]
            == "internal_subject_adaptive"
        ]

        self.assertTrue(
            adaptive[
                "dependency"
            ].astype(
                str
            ).str.startswith(
                "internal_nested_loso/"
            ).all()
        )

    def test_subject_adaptive_budgets(self):
        adaptive = self.registry.loc[
            self.registry[
                "stage"
            ]
            == "internal_subject_adaptive"
        ].copy()

        self.assertEqual(
            len(
                adaptive
            ),
            936,
        )

        budgets = sorted(
            adaptive[
                "calibration_budget_seconds_per_class"
            ]
            .dropna()
            .astype(int)
            .unique()
            .tolist()
        )

        self.assertEqual(
            budgets,
            [
                30,
                60,
                120,
            ],
        )

        counts = (
            adaptive[
                "calibration_budget_seconds_per_class"
            ]
            .astype(int)
            .value_counts()
            .sort_index()
            .to_dict()
        )

        self.assertEqual(
            counts,
            {
                30: 312,
                60: 312,
                120: 312,
            },
        )

        self.assertEqual(
            self.plan[
                "internal_subject_adaptive_budgets_seconds_per_class"
            ],
            [
                30,
                60,
                120,
            ],
        )

        self.assertEqual(
            self.plan[
                "highlighted_subject_adaptive_budget_seconds_per_class"
            ],
            120,
        )

    def test_runtime_progress_contract(self):
        progress = self.plan[
            "runtime_progress_contract"
        ]

        self.assertTrue(
            progress[
                "required"
            ]
        )

        self.assertTrue(
            progress[
                "overall_job_progress"
            ]
        )

        self.assertTrue(
            progress[
                "epoch_progress"
            ]
        )

        self.assertTrue(
            progress[
                "show_eta"
            ]
        )

        self.assertTrue(
            progress[
                "show_elapsed_time"
            ]
        )

    def test_plan_performance_blind(self):
        self.assertTrue(
            self.plan[
                "performance_blind"
            ]
        )

        self.assertFalse(
            self.plan[
                "real_model_training_performed"
            ]
        )

        self.assertFalse(
            self.plan[
                "performance_observed"
            ]
        )

    def test_planned_operations(self):
        self.assertEqual(
            self.plan[
                "planned_fit_operations"
            ][
                "total"
            ],
            3936,
        )

    def test_dry_run_firewall(self):
        self.assertTrue(
            self.dry[
                "real_research_data_loaded"
            ]
        )

        self.assertFalse(
            self.dry[
                "optimizer_created"
            ]
        )

        self.assertFalse(
            self.dry[
                "loss_computed"
            ]
        )

        self.assertFalse(
            self.dry[
                "backward_pass_performed"
            ]
        )

        self.assertFalse(
            self.dry[
                "parameter_update_performed"
            ]
        )

        self.assertFalse(
            self.dry[
                "performance_metrics_computed"
            ]
        )

        self.assertFalse(
            self.dry[
                "performance_observed"
            ]
        )

    def test_bbbd_window_parity(self):
        bbbd = self.dry[
            "bbbd"
        ]

        self.assertTrue(
            bbbd[
                "locked_candidate_window_parity"
            ]
        )

        self.assertTrue(
            bbbd[
                "locked_accepted_window_parity"
            ]
        )

    def test_pupil_contract(self):
        self.assertEqual(
            self.dry[
                "internal"
            ][
                "pupil_model_input"
            ],
            "r",
        )

        self.assertEqual(
            self.dry[
                "bbbd"
            ][
                "pupil_model_input"
            ],
            "pupil_size",
        )


if __name__ == "__main__":
    unittest.main()
