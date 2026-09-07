
import unittest

import torch

from src.models.deep_learning_baselines import (
    LOCKED_MODALITY_PATHS,
    CausalConv1d,
    MultibranchTCN,
    ShallowConvNet,
    count_trainable_parameters,
)


class TestShallowConvNet(unittest.TestCase):

    def test_four_second_shape(self):
        model = ShallowConvNet(
            n_classes=3,
            n_times=512,
        )

        x = torch.randn(
            3,
            8,
            512,
        )

        output = model(x)

        self.assertEqual(
            tuple(output.shape),
            (3, 3),
        )

    def test_all_locked_internal_durations(self):
        for seconds in [
            8,
            16,
            24,
            32,
        ]:
            with self.subTest(
                seconds=seconds
            ):
                model = ShallowConvNet(
                    n_classes=3,
                    n_times=128 * seconds,
                )

                x = torch.randn(
                    2,
                    8,
                    128 * seconds,
                )

                output = model(x)

                self.assertEqual(
                    tuple(output.shape),
                    (2, 3),
                )

    def test_binary_head(self):
        model = ShallowConvNet(
            n_classes=2,
            n_times=512,
        )

        output = model(
            torch.randn(
                2,
                8,
                512,
            )
        )

        self.assertEqual(
            tuple(output.shape),
            (2, 2),
        )

    def test_reject_wrong_eeg_channel_count(self):
        model = ShallowConvNet(
            n_classes=3,
            n_times=512,
        )

        with self.assertRaises(
            ValueError
        ):
            model(
                torch.randn(
                    2,
                    7,
                    512,
                )
            )

    def test_parameter_count_positive(self):
        model = ShallowConvNet(
            n_classes=3,
            n_times=512,
        )

        self.assertGreater(
            count_trainable_parameters(
                model
            ),
            0,
        )


class TestCausalConvolution(unittest.TestCase):

    def test_future_input_does_not_change_past_output(self):
        torch.manual_seed(42)

        layer = CausalConv1d(
            2,
            4,
            kernel_size=3,
            dilation=2,
        )

        layer.eval()

        x1 = torch.randn(
            1,
            2,
            32,
        )

        x2 = x1.clone()

        # Modify only future samples.
        x2[
            :,
            :,
            20:
        ] = (
            x2[
                :,
                :,
                20:
            ]
            + 100.0
        )

        with torch.no_grad():
            y1 = layer(x1)
            y2 = layer(x2)

        self.assertTrue(
            torch.allclose(
                y1[
                    :,
                    :,
                    :20
                ],
                y2[
                    :,
                    :,
                    :20
                ],
                atol=1.0e-6,
                rtol=1.0e-6,
            )
        )


class TestMultibranchTCN(unittest.TestCase):

    def _inputs(self):
        return {
            "EEG":
                torch.randn(
                    2,
                    8,
                    512,
                ),

            "ECG":
                torch.randn(
                    2,
                    1,
                    512,
                ),

            "Pupil":
                torch.randn(
                    2,
                    1,
                    120,
                ),
        }

    def test_all_seven_paths(self):
        inputs = self._inputs()

        self.assertEqual(
            len(
                LOCKED_MODALITY_PATHS
            ),
            7,
        )

        for path in (
            LOCKED_MODALITY_PATHS
        ):
            with self.subTest(
                path=path
            ):
                model = MultibranchTCN(
                    path=path,
                    n_classes=3,
                )

                output = model(
                    inputs
                )

                self.assertEqual(
                    tuple(output.shape),
                    (2, 3),
                )

                self.assertGreater(
                    count_trainable_parameters(
                        model
                    ),
                    0,
                )

    def test_binary_bbbd_head(self):
        model = MultibranchTCN(
            path="ECG_EEG_Pupil",
            n_classes=2,
        )

        output = model(
            self._inputs()
        )

        self.assertEqual(
            tuple(output.shape),
            (2, 2),
        )

    def test_native_temporal_lengths_can_differ(self):
        model = MultibranchTCN(
            path="ECG_EEG_Pupil",
            n_classes=3,
        )

        inputs = {
            "EEG":
                torch.randn(
                    2,
                    8,
                    2048,
                ),

            "ECG":
                torch.randn(
                    2,
                    1,
                    2048,
                ),

            "Pupil":
                torch.randn(
                    2,
                    1,
                    480,
                ),
        }

        output = model(
            inputs
        )

        self.assertEqual(
            tuple(output.shape),
            (2, 3),
        )

    def test_three_channel_pupil_input_is_rejected(self):
        model = MultibranchTCN(
            path="Pupil",
            n_classes=3,
        )

        with self.assertRaises(
            ValueError
        ):
            model(
                {
                    "Pupil":
                        torch.randn(
                            2,
                            3,
                            120,
                        )
                }
            )

    def test_missing_modality_is_rejected(self):
        model = MultibranchTCN(
            path="ECG_EEG",
            n_classes=3,
        )

        with self.assertRaises(
            ValueError
        ):
            model(
                {
                    "EEG":
                        torch.randn(
                            2,
                            8,
                            512,
                        )
                }
            )


if __name__ == "__main__":
    unittest.main()
