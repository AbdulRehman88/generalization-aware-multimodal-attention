from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.evaluation.internal_calibrated_temporal_pilot import (
    assign_protocol_roles,
    build_causal_context,
    robust_baseline_normalize,
)


class CalibratedTemporalPilotTests(
    unittest.TestCase
):
    def test_baseline_normalization_centers_participant_baseline(
        self,
    ) -> None:
        rows = []

        for participant_offset, participant in enumerate(
            ["P01", "P02"]
        ):
            for phase in (1, 2, 3):
                for window_index in range(5):
                    rows.append(
                        {
                            "segment_id":
                                f"{participant}_{phase}_{window_index}",
                            "participant": participant,
                            "phase": phase,
                            "label": phase - 1,
                            "window_index": window_index,
                            "feature_a":
                                participant_offset * 100
                                + phase * 10
                                + window_index,
                            "feature_b":
                                participant_offset * 50
                                + phase
                                + window_index * 2,
                        }
                    )

        frame = pd.DataFrame(rows)

        normalized, _ = robust_baseline_normalize(
            frame,
            ["feature_a", "feature_b"],
            calibration_windows_per_phase=5,
        )

        for participant in ("P01", "P02"):
            baseline = normalized.loc[
                (
                    normalized["participant"]
                    == participant
                )
                & (
                    normalized["phase"]
                    == 1
                ),
                [
                    "feature_a",
                    "feature_b",
                ],
            ]

            np.testing.assert_allclose(
                np.median(
                    baseline.to_numpy(),
                    axis=0,
                ),
                np.zeros(2),
                atol=1e-12,
            )

    def test_context_never_crosses_phase_boundary(
        self,
    ) -> None:
        rows = []

        for phase in (1, 2):
            for window_index in range(6):
                rows.append(
                    {
                        "segment_id":
                            f"P01_{phase}_{window_index}",
                        "participant": "P01",
                        "phase": phase,
                        "label": phase,
                        "window_index": window_index,
                        "feature":
                            phase * 100
                            + window_index,
                    }
                )

        context = build_causal_context(
            pd.DataFrame(rows),
            ["feature"],
            context_windows=3,
        )

        self.assertEqual(
            len(context),
            8,
        )

        phase_one_mean = context.loc[
            (
                context["phase"] == 1
            )
            & (
                context["window_index"] == 2
            ),
            "context_mean__feature",
        ].iloc[0]

        phase_two_mean = context.loc[
            (
                context["phase"] == 2
            )
            & (
                context["window_index"] == 2
            ),
            "context_mean__feature",
        ].iloc[0]

        self.assertEqual(
            phase_one_mean,
            101.0,
        )

        self.assertEqual(
            phase_two_mean,
            201.0,
        )

    def test_protocol_roles_include_guard_interval(
        self,
    ) -> None:
        context = pd.DataFrame(
            {
                "window_index":
                    np.arange(4, 25),
            }
        )

        roles = assign_protocol_roles(
            context,
            calibration_windows_per_phase=15,
            guard_windows=5,
        )

        self.assertEqual(
            int(
                (
                    roles
                    == "calibration"
                ).sum()
            ),
            11,
        )

        self.assertEqual(
            int(
                (
                    roles
                    == "guard"
                ).sum()
            ),
            5,
        )

        self.assertEqual(
            int(
                (
                    roles
                    == "test"
                ).sum()
            ),
            5,
        )


if __name__ == "__main__":
    unittest.main()