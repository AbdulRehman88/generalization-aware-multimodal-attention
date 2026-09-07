"""Prespecified ds003838 temporal-window representations."""

from __future__ import annotations

import copy
from typing import Any


EXPECTED_WINDOW_NAMES = [
    "late_encoding",
    "immediate_retention",
    "delayed_retention",
]


def temporal_window_protocol(
    config: dict[str, Any],
) -> dict[str, Any]:
    """Load and validate the prespecified temporal-window protocol."""

    protocol = config[
        "training"
    ][
        "ds003838_temporal_windows"
    ]

    baseline = protocol[
        "baseline"
    ]

    candidates = protocol[
        "candidates"
    ]

    names = [
        str(candidate["name"])
        for candidate in candidates
    ]

    if names != EXPECTED_WINDOW_NAMES:
        raise ValueError(
            f"Temporal-window order changed: {names}"
        )

    if len(set(names)) != len(names):
        raise ValueError(
            "Temporal-window names must be unique."
        )

    if str(
        baseline["anchor"]
    ) != "first_digit_onset":
        raise ValueError(
            "Baseline must use first-digit onset."
        )

    if float(
        baseline["start_offset_seconds"]
    ) != -4.5:
        raise ValueError(
            "Baseline start offset changed."
        )

    if float(
        baseline["duration_seconds"]
    ) != 4.0:
        raise ValueError(
            "Baseline duration changed."
        )

    expected = {
        "late_encoding": (
            "final_digit_offset",
            -4.0,
            4.0,
        ),
        "immediate_retention": (
            "final_digit_offset",
            0.0,
            4.0,
        ),
        "delayed_retention": (
            "final_digit_offset",
            1.0,
            4.0,
        ),
    }

    for candidate in candidates:
        name = str(
            candidate["name"]
        )

        observed = (
            str(candidate["anchor"]),
            float(
                candidate[
                    "start_offset_seconds"
                ]
            ),
            float(
                candidate[
                    "duration_seconds"
                ]
            ),
        )

        if observed != expected[name]:
            raise ValueError(
                f"{name}: temporal definition changed: {observed}"
            )

    return protocol


def build_candidate_config(
    config: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    """Build one isolated post-task window configuration."""

    result = copy.deepcopy(
        config
    )

    window = result[
        "datasets"
    ][
        "ds003838"
    ][
        "window"
    ]

    window["anchor"] = str(
        candidate["anchor"]
    )

    window[
        "start_offset_seconds"
    ] = float(
        candidate[
            "start_offset_seconds"
        ]
    )

    window[
        "duration_seconds"
    ] = float(
        candidate[
            "duration_seconds"
        ]
    )

    return result