
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.models.deep_learning_training_runtime import (
    ChannelZScoreNormalizer,
    amp_enabled_for_device,
    build_frozen_model,
    build_seeded_frozen_model,
    configure_final_head_only,
    expand_execution_registry,
    inverse_frequency_class_weights,
    required_checkpoint_keys,
)
import pandas as pd
from src.models.deep_learning_training_runtime import REGISTRY_V2



class TestDeepLearningTrainingRuntime(unittest.TestCase):

    def test_registry_expands_to_frozen_operation_count(self):

        operations = expand_execution_registry()

        self.assertEqual(
            len(
                operations
            ),
            3936,
        )

        self.assertEqual(
            operations[
                "stage"
            ]
            .value_counts()
            .sort_index()
            .to_dict(),
            {
                "bbbd_cross_experiment":
                    48,

                "bbbd_within_experiment":
                    864,

                "internal_conventional":
                    840,

                "internal_nested_loso":
                    1248,

                "internal_subject_adaptive":
                    936,
            },
        )


    def test_all_operation_ids_are_globally_unique(self):

        operations = expand_execution_registry()

        self.assertEqual(
            len(
                operations
            ),
            3936,
        )

        self.assertEqual(
            operations[
                "operation_id"
            ].nunique(),
            3936,
        )

        self.assertFalse(
            operations[
                "operation_id"
            ].duplicated().any()
        )



    def test_conventional_operation_ids_include_protocol(self):

        operations = expand_execution_registry()

        conventional = operations.loc[
            operations[
                "stage"
            ]
            == "internal_conventional"
        ].copy()

        protocols = sorted(
            conventional[
                "protocol"
            ].astype(str).unique().tolist()
        )

        self.assertEqual(
            protocols,
            [
                "legacy_repeated_stratified_window_5x3",
                "legacy_shuffle_split_window",
                "legacy_stratified_window_5fold",
                "primary_grouped_5fold",
            ],
        )

        for protocol in protocols:

            subset = conventional.loc[
                conventional[
                    "protocol"
                ].astype(str)
                == protocol
            ]

            self.assertTrue(
                subset[
                    "operation_id"
                ].astype(str).str.contains(
                    f"/{protocol}/",
                    regex=False,
                ).all()
            )


    def test_seeded_model_initialization_is_reproducible(self):

        first = build_seeded_frozen_model(
            training_seed=3407,
            model_name="ShallowConvNet",
            path="EEG",
            n_classes=3,
            n_times=128,
        )

        second = build_seeded_frozen_model(
            training_seed=3407,
            model_name="ShallowConvNet",
            path="EEG",
            n_classes=3,
            n_times=128,
        )

        first_state = first.state_dict()
        second_state = second.state_dict()

        self.assertEqual(
            first_state.keys(),
            second_state.keys(),
        )

        for key in first_state:

            self.assertTrue(
                torch.equal(
                    first_state[
                        key
                    ],
                    second_state[
                        key
                    ],
                ),
                key,
            )


    def test_nested_durations(self):

        operations = expand_execution_registry()

        nested = operations.loc[
            operations[
                "stage"
            ]
            == "internal_nested_loso"
        ]

        self.assertEqual(
            sorted(
                nested[
                    "duration_candidate_seconds"
                ]
                .astype(int)
                .unique()
                .tolist()
            ),
            [
                8,
                16,
                24,
                32,
            ],
        )

    def test_adaptive_budgets(self):

        operations = expand_execution_registry()

        adaptive = operations.loc[
            operations[
                "stage"
            ]
            == "internal_subject_adaptive"
        ].copy()

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
                30:
                    312,

                60:
                    312,

                120:
                    312,
            },
        )


    def test_operation_ids_unique(self):

        operations = expand_execution_registry()

        self.assertFalse(
            operations[
                "operation_id"
            ].duplicated().any()
        )

    def test_training_only_normalizer(self):

        train = {
            "EEG":
                np.asarray(
                    [
                        [
                            [
                                1.0,
                                2.0,
                                3.0,
                            ],
                        ],
                        [
                            [
                                4.0,
                                5.0,
                                6.0,
                            ],
                        ],
                    ],
                    dtype=np.float32,
                ),
        }

        validation = {
            "EEG":
                np.full(
                    (
                        2,
                        1,
                        3,
                    ),
                    1000.0,
                    dtype=np.float32,
                ),
        }

        normalizer = ChannelZScoreNormalizer().fit(
            train
        )

        expected = np.mean(
            train[
                "EEG"
            ],
            axis=(
                0,
                2,
            ),
            dtype=np.float64,
        )

        np.testing.assert_allclose(
            normalizer.statistics[
                "EEG"
            ].mean,
            expected,
            atol=0.0,
            rtol=0.0,
        )

        transformed = normalizer.transform(
            validation
        )

        self.assertGreater(
            float(
                np.mean(
                    transformed[
                        "EEG"
                    ]
                )
            ),
            100.0,
        )

    def test_zero_std_fallback(self):

        values = {
            "ECG":
                np.ones(
                    (
                        4,
                        1,
                        10,
                    ),
                    dtype=np.float32,
                ),
        }

        normalizer = ChannelZScoreNormalizer().fit(
            values
        )

        self.assertEqual(
            float(
                normalizer.statistics[
                    "ECG"
                ].std[
                    0
                ]
            ),
            1.0,
        )

    def test_inverse_frequency_class_weights(self):

        labels = np.asarray(
            [
                0,
                0,
                0,
                0,
                1,
                1,
                2,
            ],
            dtype=np.int64,
        )

        actual = inverse_frequency_class_weights(
            labels,
            n_classes=3,
        )

        expected = np.asarray(
            [
                7.0 / 12.0,
                7.0 / 6.0,
                7.0 / 3.0,
            ],
            dtype=np.float32,
        )

        np.testing.assert_allclose(
            actual,
            expected,
            rtol=1.0e-6,
            atol=1.0e-7,
        )

    def test_missing_training_class_fails(self):

        labels = np.asarray(
            [
                0,
                0,
                1,
                1,
            ],
            dtype=np.int64,
        )

        with self.assertRaises(
            ValueError
        ):
            inverse_frequency_class_weights(
                labels,
                n_classes=3,
            )

    def test_shallow_final_head_scope(self):

        model = build_frozen_model(
            model_name="ShallowConvNet",
            path="EEG",
            n_classes=3,
            n_times=128,
        )

        names = configure_final_head_only(
            model
        )

        self.assertEqual(
            names,
            [
                "classifier.weight",
                "classifier.bias",
            ],
        )

    def test_tcn_final_head_scope(self):

        model = build_frozen_model(
            model_name="MultibranchTCN",
            path="ECG_EEG_Pupil",
            n_classes=3,
            n_times=128,
        )

        names = configure_final_head_only(
            model
        )

        self.assertEqual(
            names,
            [
                "fusion.3.weight",
                "fusion.3.bias",
            ],
        )

    def test_cpu_amp_disabled(self):

        self.assertFalse(
            amp_enabled_for_device(
                torch.device(
                    "cpu"
                )
            )
        )

    def test_cuda_amp_rule(self):

        self.assertTrue(
            amp_enabled_for_device(
                torch.device(
                    "cuda"
                )
            )
        )

    def test_checkpoint_contract_keys(self):

        expected = {
            "job_identity",
            "protocol_revision",
            "registry_row_hash",
            "split_identity",
            "model_state_dict",
            "optimizer_state_dict",
            "scheduler_state_dict",
            "amp_scaler_state_dict_when_cuda",
            "epoch",
            "best_validation_balanced_accuracy",
            "best_validation_macro_f1",
            "best_epoch",
            "early_stopping_bad_epoch_count",
            "learning_rate",
            "training_normalizer",
            "python_random_state",
            "numpy_random_state",
            "torch_cpu_rng_state",
            "torch_cuda_rng_state_all_devices_when_cuda",
            "training_seed",
        }

        self.assertEqual(
            required_checkpoint_keys(),
            expected,
        )


    def test_proposed_v3_registry_contract_in_memory(self):

        registry = pd.read_csv(
            REGISTRY_V2,
            low_memory=False,
        )

        proposed = registry.copy()


        pupil_tcn = (
            (
                proposed[
                    "model"
                ].astype(str)
                == "MultibranchTCN"
            )
            & proposed[
                "path"
            ].astype(str).str.contains(
                "Pupil",
                regex=False,
            )
        )


        nested_pupil = (
            (
                proposed[
                    "stage"
                ].astype(str)
                == "internal_nested_loso"
            )
            & pupil_tcn
        )


        proposed.loc[
            nested_pupil,
            "candidate_durations_seconds",
        ] = "8"


        adaptive_pupil = (
            (
                proposed[
                    "stage"
                ].astype(str)
                == "internal_subject_adaptive"
            )
            & pupil_tcn
        )


        self.assertEqual(
            int(
                adaptive_pupil.sum()
            ),
            468,
        )


        proposed = (
            proposed.loc[
                ~adaptive_pupil
            ]
            .copy()
            .reset_index(
                drop=True
            )
        )


        self.assertEqual(
            len(
                proposed
            ),
            2532,
        )


        operations = expand_execution_registry(
            proposed
        )


        self.assertEqual(
            len(
                operations
            ),
            3000,
        )


        self.assertEqual(
            operations[
                "operation_id"
            ].nunique(),
            3000,
        )


        self.assertFalse(
            operations[
                "operation_id"
            ].duplicated().any()
        )


        self.assertEqual(
            operations[
                "stage"
            ]
            .value_counts()
            .sort_index()
            .to_dict(),
            {
                "bbbd_cross_experiment":
                    48,

                "bbbd_within_experiment":
                    864,

                "internal_conventional":
                    840,

                "internal_nested_loso":
                    780,

                "internal_subject_adaptive":
                    468,
            },
        )


        adaptive = operations.loc[
            operations[
                "stage"
            ]
            == "internal_subject_adaptive"
        ]


        self.assertEqual(
            adaptive[
                "calibration_budget_seconds_per_class"
            ]
            .astype(int)
            .value_counts()
            .sort_index()
            .to_dict(),
            {
                30:
                    156,

                60:
                    156,

                120:
                    156,
            },
        )


        self.assertFalse(
            adaptive[
                "path"
            ]
            .astype(str)
            .str.contains(
                "Pupil",
                regex=False,
            )
            .any()
        )


        nested_pupil_operations = operations.loc[
            (
                operations[
                    "stage"
                ]
                == "internal_nested_loso"
            )
            & (
                operations[
                    "model"
                ]
                == "MultibranchTCN"
            )
            & operations[
                "path"
            ]
            .astype(str)
            .str.contains(
                "Pupil",
                regex=False,
            )
        ]


        self.assertEqual(
            set(
                nested_pupil_operations[
                    "duration_candidate_seconds"
                ].astype(int)
            ),
            {
                8
            },
        )


if __name__ == "__main__":
    unittest.main()
