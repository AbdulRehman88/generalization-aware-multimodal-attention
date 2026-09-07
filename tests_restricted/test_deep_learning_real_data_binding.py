
import json
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

AUDIT = (
    ROOT
    / "_research_audit/"
      "deep_learning_real_data_binding_audit_v14.json"
)

SUMMARY = (
    ROOT
    / "_research_audit/"
      "deep_learning_real_data_binding_summary_v14.csv"
)


class TestDeepLearningRealDataBinding(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        if not AUDIT.is_file():
            raise RuntimeError(
                "Real-data binding audit has not been generated."
            )

        if not SUMMARY.is_file():
            raise RuntimeError(
                "Real-data binding summary has not been generated."
            )

        cls.audit = json.loads(
            AUDIT.read_text(
                encoding="utf-8"
            )
        )

        cls.summary = pd.read_csv(
            SUMMARY,
            low_memory=False,
        )

    def test_protocol_revision(self):

        self.assertEqual(
            self.audit[
                "protocol_revision"
            ],
            "1.4",
        )

    def test_binding_gate_passed(self):

        self.assertTrue(
            self.audit[
                "all_real_data_bindings_passed"
            ]
        )

    def test_frozen_registry_replay(self):

        self.assertTrue(
            self.audit[
                "frozen_registry_replay_exact"
            ]
        )

        self.assertEqual(
            self.audit[
                "frozen_registry_replay_rows"
            ],
            3000,
        )

    def test_global_operation_count(self):

        self.assertEqual(
            self.audit[
                "expanded_fit_adaptation_operations"
            ],
            3000,
        )

        self.assertEqual(
            self.audit[
                "unique_operation_ids"
            ],
            3000,
        )

    def test_adaptive_candidate_count(self):

        self.assertEqual(
            self.audit[
                "adaptive_candidate_bindings_checked"
            ],
            624,
        )

        self.assertGreater(
            self.audit[
                "adaptive_minimum_eligible_calibration_windows"
            ],
            0,
        )

        self.assertGreater(
            self.audit[
                "adaptive_minimum_eligible_test_windows"
            ],
            0,
        )

    def test_bbbd_universe(self):

        self.assertEqual(
            self.audit[
                "bbbd_participants"
            ],
            36,
        )

        self.assertEqual(
            self.audit[
                "bbbd_recordings"
            ],
            392,
        )

        self.assertEqual(
            self.audit[
                "bbbd_accepted_windows"
            ],
            51831,
        )

    def test_model_path_combinations(self):

        self.assertEqual(
            self.audit[
                "model_path_combinations"
            ],
            8,
        )

    def test_no_model_execution(self):

        for key in [
            "model_initialized",
            "optimizer_created",
            "normalization_fitted",
            "loss_computed",
            "predictions_generated",
            "neural_network_fitting_performed",
            "performance_metrics_computed",
            "performance_observed",
        ]:

            self.assertFalse(
                self.audit[
                    key
                ],
                key,
            )

    def test_real_training_gate_still_closed(self):

        self.assertFalse(
            self.audit[
                "real_training_gate_open"
            ]
        )

    def test_summary_bindable(self):

        self.assertGreater(
            len(
                self.summary
            ),
            0,
        )

        self.assertTrue(
            self.summary[
                "bindable"
            ].astype(bool).all()
        )


class TestDeepLearningRealDataBindingExactCounts(unittest.TestCase):

    def test_exact_binding_counts(self):

        audit = json.loads(
            AUDIT.read_text(
                encoding="utf-8"
            )
        )

        self.assertEqual(
            audit[
                "internal_conventional_unique_bindings"
            ],
            280,
        )

        self.assertEqual(
            audit[
                "internal_nested_unique_bindings"
            ],
            260,
        )

        self.assertEqual(
            audit[
                "adaptive_candidate_bindings_checked"
            ],
            624,
        )

        self.assertEqual(
            audit[
                "bbbd_unique_bindings"
            ],
            304,
        )

        self.assertEqual(
            audit[
                "expanded_fit_adaptation_operations"
            ],
            3000,
        )

        self.assertEqual(
            audit[
                "unique_operation_ids"
            ],
            3000,
        )


if __name__ == "__main__":
    unittest.main()
