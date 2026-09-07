from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.evaluation.audit_personalized_final import (
    participant_cluster_bootstrap,
    validate_probabilities,
    validate_selected_features,
)


class PersonalizedFinalAuditTests(
    unittest.TestCase
):
    def test_probability_validation_accepts_normalized_rows(
        self,
    ) -> None:
        frame = pd.DataFrame(
            {
                "probability_class_0":
                    [0.7, 0.1],
                "probability_class_1":
                    [0.2, 0.3],
                "probability_class_2":
                    [0.1, 0.6],
            }
        )

        validate_probabilities(
            frame,
            "synthetic",
        )

    def test_selected_features_reject_metadata(
        self,
    ) -> None:
        valid = pd.DataFrame(
            {
                "feature": [
                    "context_mean__eeg_AF7_log_power_alpha",
                    "context_std__ecg_rms",
                ]
            }
        )

        features = validate_selected_features(
            valid
        )

        self.assertEqual(
            len(features),
            2,
        )

        invalid = pd.DataFrame(
            {
                "feature": [
                    "context_mean__participant",
                ]
            }
        )

        with self.assertRaises(
            RuntimeError
        ):
            validate_selected_features(
                invalid
            )

    def test_cluster_bootstrap_is_deterministic(
        self,
    ) -> None:
        rows = []

        for participant in (
            "P01",
            "P02",
            "P03",
        ):
            for label in (
                0,
                1,
                2,
            ):
                probabilities = [
                    0.05,
                    0.05,
                    0.05,
                ]

                probabilities[
                    label
                ] = 0.90

                rows.append(
                    {
                        "participant":
                            participant,
                        "true_label":
                            label,
                        "probability_class_0":
                            probabilities[0],
                        "probability_class_1":
                            probabilities[1],
                        "probability_class_2":
                            probabilities[2],
                    }
                )

        frame = pd.DataFrame(
            rows
        )

        first = participant_cluster_bootstrap(
            frame,
            iterations=10,
            random_seed=7,
        )

        second = participant_cluster_bootstrap(
            frame,
            iterations=10,
            random_seed=7,
        )

        pd.testing.assert_frame_equal(
            first,
            second,
        )

        np.testing.assert_allclose(
            first["accuracy"],
            np.ones(10),
        )


if __name__ == "__main__":
    unittest.main()