
import json
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

AUDIT = (
    ROOT
    / "_research_audit/deep_learning_raw_cache_audit_v1.json"
)

MANIFEST = (
    ROOT
    / "_research_audit/deep_learning_raw_cache_manifest_v1.csv"
)


class TestDeepLearningRawCache(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        if not AUDIT.is_file():
            raise RuntimeError(
                "Deep-learning raw-cache audit has not been generated."
            )

        if not MANIFEST.is_file():
            raise RuntimeError(
                "Deep-learning raw-cache manifest has not been generated."
            )

        cls.audit = json.loads(
            AUDIT.read_text(
                encoding="utf-8"
            )
        )

        cls.manifest = pd.read_csv(
            MANIFEST,
            low_memory=False,
        )

    def test_protocol_revision(self):
        self.assertEqual(
            self.audit[
                "protocol_revision"
            ],
            "1.2",
        )

    def test_no_training_or_performance(self):
        for key in [
            "normalization_statistics_fitted",
            "optimizer_created",
            "loss_computed",
            "backward_pass_performed",
            "neural_network_fitting_performed",
            "predictions_generated",
            "performance_metrics_computed",
            "performance_observed",
        ]:
            self.assertFalse(
                self.audit[
                    key
                ]
            )

    def test_internal_counts(self):
        expected = {
            "4":
                (
                    3315,
                    2306,
                ),

            "8":
                (
                    3276,
                    1962,
                ),

            "16":
                (
                    1599,
                    767,
                ),

            "24":
                (
                    1066,
                    433,
                ),

            "32":
                (
                    767,
                    267,
                ),
        }

        for duration, (
            candidates,
            pupil_valid,
        ) in expected.items():

            with self.subTest(
                duration=duration
            ):

                item = self.audit[
                    "internal"
                ][
                    duration
                ]

                self.assertEqual(
                    item[
                        "candidate_windows"
                    ],
                    candidates,
                )

                self.assertEqual(
                    item[
                        "pupil_valid_windows"
                    ],
                    pupil_valid,
                )

    def test_bbbd_counts(self):
        bbbd = self.audit[
            "bbbd"
        ]

        self.assertEqual(
            bbbd[
                "participants"
            ],
            36,
        )

        self.assertEqual(
            bbbd[
                "recordings"
            ],
            392,
        )

        self.assertEqual(
            bbbd[
                "candidate_windows"
            ],
            53512,
        )

        self.assertEqual(
            bbbd[
                "accepted_windows"
            ],
            51831,
        )

        self.assertEqual(
            bbbd[
                "rejected_pupil_windows"
            ],
            1681,
        )

    def test_pupil_semantics(self):
        self.assertEqual(
            self.audit[
                "bbbd"
            ][
                "pupil_semantics"
            ],
            "pupil_size only",
        )

    def test_cache_manifest_size(self):
        # Five internal cache files + 392 BBBD recording caches.
        self.assertEqual(
            len(
                self.manifest
            ),
            397,
        )

    def test_all_cache_files_exist(self):
        cache_paths = [
            ROOT / path_string
            for path_string in self.manifest[
                "cache_path"
            ].astype(str)
        ]

        existing_paths = [
            path
            for path in cache_paths
            if path.is_file()
        ]

        if not existing_paths:
            self.skipTest(
                "Generated raw-signal caches are not distributed "
                "with the source repository."
            )

        missing_paths = [
            path
            for path in cache_paths
            if not path.is_file()
        ]

        self.assertFalse(
            missing_paths,
            (
                "The generated cache is partially present: "
                f"{len(missing_paths)} of {len(cache_paths)} "
                "files are missing. First missing file: "
                f"{missing_paths[0] if missing_paths else 'none'}"
            ),
        )

    def test_checksum_gate(self):
        self.assertTrue(
            self.audit[
                "cache_file_checksums_verified"
            ]
        )

        self.assertTrue(
            self.audit[
                "full_universe_parity_passed"
            ]
        )

    def test_progress_display_used(self):
        self.assertTrue(
            self.audit[
                "execution_progress_display"
            ]
        )


if __name__ == "__main__":
    unittest.main()
