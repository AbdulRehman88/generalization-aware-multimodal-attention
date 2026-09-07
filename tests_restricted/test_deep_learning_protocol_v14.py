import json
import unittest
from pathlib import Path

import pandas as pd
import yaml

from src.models.deep_learning_training_runtime import (
    expand_execution_registry,
)


ROOT = Path(__file__).resolve().parents[1]


class TestDeepLearningProtocolV14(unittest.TestCase):

    def test_protocol_revision_and_gate(self):

        protocol = yaml.safe_load(
            (
                ROOT
                / "configs/deep_learning_baselines.yaml"
            ).read_text(
                encoding="utf-8"
            )
        )

        self.assertEqual(
            str(
                protocol[
                    "protocol_revision"
                ]
            ),
            "1.4",
        )

        self.assertFalse(
            protocol[
                "execution_gate"
            ][
                "real_training_allowed_now"
            ]
        )

    def test_registry_v3_contract(self):

        registry = pd.read_csv(
            ROOT
            / "artifacts/revision/manifests/"
              "deep_learning_execution_registry_v3.csv",
            low_memory=False,
        )

        self.assertEqual(
            len(
                registry
            ),
            2532,
        )

        operations = expand_execution_registry(
            registry
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

        adaptive = operations.loc[
            operations[
                "stage"
            ]
            == "internal_subject_adaptive"
        ]

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

        nested_pupil = operations.loc[
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
                nested_pupil[
                    "duration_candidate_seconds"
                ].astype(int)
            ),
            {
                8
            },
        )

    def test_final_binding_audit(self):

        audit = json.loads(
            (
                ROOT
                / "_research_audit/"
                  "deep_learning_real_data_binding_audit_v14.json"
            ).read_text(
                encoding="utf-8"
            )
        )

        self.assertTrue(
            audit[
                "all_bindings_passed"
            ]
        )

        self.assertEqual(
            audit[
                "nested_structural_bindings"
            ],
            260,
        )

        self.assertEqual(
            audit[
                "adaptive_structural_checks"
            ],
            624,
        )

        self.assertEqual(
            audit[
                "final_expanded_operations"
            ],
            3000,
        )

        self.assertFalse(
            audit[
                "training_performed"
            ]
        )

        self.assertFalse(
            audit[
                "performance_observed"
            ]
        )


if __name__ == "__main__":
    unittest.main()
