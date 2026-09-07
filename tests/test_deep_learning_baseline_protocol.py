
import json
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]

CONFIG = (
    ROOT
    / "configs/deep_learning_baselines.yaml"
)

AUDIT = (
    ROOT
    / "_research_audit/deep_learning_baseline_protocol_v1.json"
)


class TestDeepLearningBaselineProtocol(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.config = yaml.safe_load(
            CONFIG.read_text(
                encoding="utf-8"
            )
        )

        cls.audit = json.loads(
            AUDIT.read_text(
                encoding="utf-8"
            )
        )

    def test_protocol_identity(self):
        self.assertEqual(
            self.config[
                "protocol_identity"
            ],
            "deep_learning_baselines_v1",
        )

    def test_preperformance_lock(self):
        self.assertEqual(
            self.config["status"],
            "frozen_before_any_deep_model_performance",
        )

        self.assertFalse(
            self.config[
                "execution_gate"
            ][
                "real_training_allowed_now"
            ]
        )

    def test_fixed_models(self):
        models = self.config[
            "models"
        ]

        self.assertEqual(
            models[
                "shallowconvnet"
            ][
                "eligible_paths"
            ],
            ["EEG"],
        )

        self.assertFalse(
            models[
                "shallowconvnet"
            ][
                "multimodal_adaptation"
            ]
        )

        self.assertEqual(
            len(
                models[
                    "multibranch_tcn"
                ][
                    "eligible_paths"
                ]
            ),
            7,
        )

    def test_internal_participants(self):
        internal = self.config[
            "internal_input"
        ]

        self.assertEqual(
            internal[
                "participant_count"
            ],
            13,
        )

        self.assertEqual(
            len(
                internal[
                    "participants"
                ]
            ),
            13,
        )

    def test_eeg_contract(self):
        eeg = self.config[
            "internal_input"
        ][
            "eeg"
        ]

        self.assertEqual(
            eeg[
                "sampling_rate_hz"
            ],
            128.0,
        )

        self.assertEqual(
            eeg[
                "channels"
            ],
            [
                "AF7",
                "Fp1",
                "Fpz",
                "Fp2",
                "AF8",
                "O1",
                "POz",
                "O2",
            ],
        )

    def test_exact_four_conventional_protocols(self):
        conventional = self.config[
            "internal_conventional_evaluation"
        ]

        self.assertEqual(
            conventional[
                "protocol_count"
            ],
            4,
        )

        protocols = conventional[
            "protocols"
        ]

        self.assertEqual(
            protocols[
                "primary_grouped_5fold"
            ][
                "n_splits"
            ],
            5,
        )

        self.assertEqual(
            protocols[
                "legacy_stratified_window_5fold"
            ][
                "n_splits"
            ],
            5,
        )

        repeated = protocols[
            "legacy_repeated_stratified_window_5x3"
        ]

        self.assertEqual(
            repeated[
                "n_splits"
            ],
            5,
        )

        self.assertEqual(
            repeated[
                "n_repeats"
            ],
            3,
        )

        shuffle = protocols[
            "legacy_shuffle_split_window"
        ]

        self.assertEqual(
            shuffle[
                "n_splits"
            ],
            10,
        )

        self.assertAlmostEqual(
            shuffle[
                "test_size"
            ],
            0.2,
        )

        self.assertFalse(
            conventional[
                "bootstrap_training_protocol_included"
            ]
        )

    def test_nested_loso(self):
        loso = self.config[
            "internal_primary_nested_loso"
        ]

        self.assertEqual(
            loso[
                "outer_folds"
            ],
            13,
        )

        self.assertEqual(
            loso[
                "inner_validation_participants_per_fold"
            ],
            2,
        )

        self.assertTrue(
            loso[
                "calibration_free"
            ]
        )

        self.assertEqual(
            loso[
                "candidate_raw_window_seconds"
            ],
            [
                8,
                16,
                24,
                32,
            ],
        )

    def test_subject_adaptive_contract(self):
        adaptive = self.config[
            "internal_120s_subject_adaptive"
        ]

        self.assertEqual(
            adaptive[
                "calibration_budget_seconds_per_class"
            ],
            120,
        )

        self.assertFalse(
            adaptive[
                "calibration_free"
            ]
        )

        self.assertTrue(
            adaptive[
                "adaptation"
            ][
                "backbone_frozen"
            ]
        )

    def test_bbbd_contract(self):
        bbbd = self.config[
            "bbbd"
        ]

        self.assertEqual(
            bbbd[
                "primary_window_seconds"
            ],
            4.0,
        )

        self.assertEqual(
            bbbd[
                "cohort"
            ][
                "participants"
            ],
            36,
        )

        self.assertFalse(
            bbbd[
                "deep_temporal_sensitivity_8_16_24_32"
            ]
        )

    def test_three_fixed_seeds(self):
        self.assertEqual(
            self.config[
                "randomness"
            ][
                "training_seeds"
            ],
            [
                42,
                3407,
                2026,
            ],
        )

    def test_bootstrap_repetitions(self):
        self.assertEqual(
            self.config[
                "reporting"
            ][
                "confidence_intervals"
            ][
                "repetitions"
            ],
            5000,
        )

    def test_harmonized_pupil_model_input(self):
        pupil = self.config[
            "internal_input"
        ][
            "pupil"
        ]

        self.assertEqual(
            pupil[
                "source_channel_count"
            ],
            3,
        )

        self.assertEqual(
            pupil[
                "model_input_channel_count"
            ],
            1,
        )

        self.assertEqual(
            pupil[
                "model_input_channel"
            ],
            "r",
        )

        self.assertEqual(
            pupil[
                "model_input_channel_index"
            ],
            2,
        )

        self.assertFalse(
            pupil[
                "gaze_coordinates_used_by_deep_model"
            ]
        )

        self.assertEqual(
            self.config[
                "models"
            ][
                "multibranch_tcn"
            ][
                "branch_input_channels"
            ][
                "Pupil"
            ],
            1,
        )

        self.assertEqual(
            self.config[
                "bbbd"
            ][
                "pupil_model_input"
            ][
                "channel_count"
            ],
            1,
        )

    def test_subject_adaptive_budget_and_progress_contract(self):
        self.assertEqual(
            self.config[
                "protocol_revision"
            ],
            "1.4",
        )

        historical = self.config[
            "internal_120s_subject_adaptive"
        ]

        self.assertFalse(
            historical[
                "active_for_new_deep_execution"
            ]
        )

        self.assertEqual(
            historical[
                "superseded_by"
            ],
            "internal_subject_adaptive",
        )

        active = self.config[
            "internal_subject_adaptive"
        ]

        self.assertTrue(
            active[
                "active_for_new_deep_execution"
            ]
        )

        self.assertEqual(
            active[
                "calibration_budgets_seconds_per_class"
            ],
            [
                30,
                60,
                120,
            ],
        )

        self.assertEqual(
            active[
                "highlighted_budget_seconds_per_class"
            ],
            120,
        )

        self.assertFalse(
            active[
                "budget_selection_by_test_performance"
            ]
        )

        progress = self.config[
            "execution_reporting"
        ][
            "progress_bar"
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

    def test_json_and_yaml_match(self):
        self.assertEqual(
            self.config,
            self.audit,
        )


if __name__ == "__main__":
    unittest.main()
