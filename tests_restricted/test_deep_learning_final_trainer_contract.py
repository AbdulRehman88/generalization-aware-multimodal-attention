
import json
import math
import unittest
from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]

CONFIG = (
    ROOT
    / "configs/deep_learning_baselines.yaml"
)

CONTRACT = (
    ROOT
    / "_research_audit/deep_learning_final_trainer_contract_v1.json"
)

ADAPTIVE_AUDIT = (
    ROOT
    / "_research_audit/"
      "deep_learning_calibration_test_split_audit_v1.csv"
)


class TestFinalDeepLearningTrainerContract(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        cls.config = yaml.safe_load(
            CONFIG.read_text(
                encoding="utf-8"
            )
        )

        cls.contract = json.loads(
            CONTRACT.read_text(
                encoding="utf-8"
            )
        )

        cls.audit = pd.read_csv(
            ADAPTIVE_AUDIT,
            low_memory=False,
        )

    def test_protocol_revision(self):

        self.assertEqual(
            self.config[
                "protocol_revision"
            ],
            "1.4",
        )

    def test_numeric_label_mapping(self):

        internal = self.config[
            "internal_input"
        ]

        self.assertEqual(
            internal[
                "numeric_class_ids"
            ],
            [
                0,
                1,
                2,
            ],
        )

        self.assertEqual(
            internal[
                "phase_to_numeric_label"
            ],
            {
                "1":
                    0,

                "2":
                    2,

                "3":
                    1,
            },
        )

        self.assertEqual(
            internal[
                "class_mapping"
            ],
            {
                "0":
                    "class_0",

                "1":
                    "class_1",

                "2":
                    "class_2",
            },
        )

    def test_adaptive_formula_against_all_rows(self):

        self.assertEqual(
            len(
                self.audit
            ),
            1053,
        )

        for row in self.audit.itertuples(
            index=False
        ):

            duration = int(
                row.duration_seconds
            )

            budget = int(
                row.calibration_budget_seconds
            )

            stride = (
                duration
                / 2.0
            )

            count = max(
                1,
                math.ceil(
                    budget
                    / stride
                ),
            )

            raw_end = (
                duration
                + (
                    count
                    - 1
                )
                * stride
            )

            threshold = (
                raw_end
                + duration
            )

            self.assertEqual(
                int(
                    row.calibration_windows
                ),
                count,
            )

            self.assertAlmostEqual(
                float(
                    row.calibration_raw_end_seconds
                ),
                raw_end,
            )

            self.assertAlmostEqual(
                float(
                    row.guard_seconds
                ),
                float(
                    duration
                ),
            )

            self.assertAlmostEqual(
                float(
                    row.test_start_threshold_seconds
                ),
                threshold,
            )

    def test_amp_is_cuda_only(self):

        amp = self.config[
            "optimization"
        ][
            "mixed_precision"
        ]

        self.assertTrue(
            amp[
                "enabled_on_cuda"
            ]
        )

        self.assertFalse(
            amp[
                "enabled_on_cpu"
            ]
        )

        self.assertEqual(
            amp[
                "cuda_autocast_dtype"
            ],
            "float16",
        )

        self.assertEqual(
            amp[
                "cpu_training_dtype"
            ],
            "float32",
        )

    def test_training_only_normalization(self):

        policy = self.config[
            "training_normalization"
        ]

        self.assertEqual(
            policy[
                "statistics_scope"
            ],
            "training partition only",
        )

        self.assertFalse(
            policy[
                "validation_used_to_fit_statistics"
            ]
        )

        self.assertFalse(
            policy[
                "test_used_to_fit_statistics"
            ]
        )

        self.assertFalse(
            policy[
                "outer_subject_calibration_used_for_global_statistics"
            ]
        )

    def test_checkpoint_resume_required(self):

        policy = self.config[
            "checkpoint_resume"
        ]

        self.assertTrue(
            policy[
                "required"
            ]
        )

        self.assertTrue(
            policy[
                "atomic_write"
            ]
        )

        self.assertTrue(
            policy[
                "resume_default"
            ]
        )

        self.assertEqual(
            policy[
                "resume_mismatch_policy"
            ],
            "hard_fail",
        )

        self.assertIn(
            "torch_cpu_rng_state",
            policy[
                "required_checkpoint_state"
            ],
        )

        self.assertIn(
            "training_normalizer",
            policy[
                "required_checkpoint_state"
            ],
        )

    def test_progress_and_eta_required(self):

        progress = self.config[
            "execution_reporting"
        ][
            "progress_bar"
        ]

        for key in [
            "required",
            "overall_job_progress",
            "epoch_progress",
            "show_elapsed_time",
            "show_eta",
            "checkpoint_resume_status",
            "show_current_learning_rate",
            "show_early_stopping_patience",
            "show_best_validation_metric",
        ]:

            self.assertTrue(
                progress[
                    key
                ],
                key,
            )

    def test_training_gate_remains_closed(self):

        gate = self.config[
            "execution_gate"
        ]

        self.assertFalse(
            gate[
                "real_training_allowed_now"
            ]
        )

        self.assertTrue(
            gate[
                "training_runner_implementation_required"
            ]
        )

        self.assertTrue(
            gate[
                "synthetic_training_smoke_required"
            ]
        )

    def test_contract_is_preperformance(self):

        self.assertFalse(
            self.contract[
                "real_training_before_contract"
            ]
        )

        self.assertFalse(
            self.contract[
                "deep_performance_observed_before_contract"
            ]
        )

        self.assertFalse(
            self.contract[
                "real_training_allowed_after_contract_only"
            ]
        )


if __name__ == "__main__":
    unittest.main()
