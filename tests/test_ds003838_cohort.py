from __future__ import annotations

import unittest

import pandas as pd

from src.preprocessing.ds003838_cohort import (
    boolean_series,
)


class DS003838CohortTests(unittest.TestCase):
    def test_boolean_series_parses_strings_safely(self) -> None:
        observed = boolean_series(
            pd.Series(
                ["True", "False", "1", "0"]
            )
        ).tolist()

        self.assertEqual(
            observed,
            [True, False, True, False],
        )

    def test_boolean_series_rejects_unknown_values(self) -> None:
        with self.assertRaises(ValueError):
            boolean_series(
                pd.Series(["True", "unknown"])
            )


if __name__ == "__main__":
    unittest.main()