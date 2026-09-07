"""Retired legacy utility retained to prevent accidental methodological misuse.

The former implementation loaded models fitted outside each held-out participant
and then reported participant-wise accuracy. It therefore did not implement
leave-one-subject-out cross-validation, despite its historical filename.

Use ``src/evaluation/internal_nested_loso_long_windows.py`` for the manuscript's
primary strict, calibration-free, nested-LOSO evaluation.
"""

from __future__ import annotations

import sys


AUTHORITATIVE_LOSO_ENTRYPOINT = (
    "src/evaluation/internal_nested_loso_long_windows.py"
)


def main() -> int:
    """Refuse execution and direct users to the validated LOSO pipeline."""
    print(
        "This legacy helper is retired because it did not perform fold-specific "
        "LOSO training and feature selection.\n"
        f"Use {AUTHORITATIVE_LOSO_ENTRYPOINT} instead.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())