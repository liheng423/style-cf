import os
import unittest
import numpy as np

from src.models.model_trainer import build_style_dataset
from src.models.utils import load_zen_data
from src.models.configs import stylecf_data_config
from src.schema import CFNAMES as CF
from src.stylecf.schema import TensorNames


class TestStyleAgent(unittest.TestCase):

    def _small_dataset(self):

        data_path = "F:\DATA\ZenTraffic\ZenTraffic30kalman.npy"
        if not os.path.exists(data_path):
            self.skipTest(f"Missing dataset: {data_path}")

        d = load_zen_data(data_path, rise=True, in_kph=False, kilo_norm=True)
        return d.head(300)

    # def test_build_style_dataset(self):
    #     d = self._small_dataset()
    #     d_filters = [lambda: np.ones(d.data.shape[0], dtype=bool)]
    #     d_filter_config = {}

    #     result = build_style_dataset(d, d_filters, d_filter_config)

    #     self.assertIn(CF.TIME, result.names)
    #     self.assertIn(CF.REACT, result.names)
    #     self.assertIn(CF.THW, result.names)

    def test_style_dataloader_output(self):
        d = self._small_dataset()
        d_filters = [lambda: np.ones(d.data.shape[0], dtype=bool)]
        d_filter_config = {}

        _, train_loader, _, _ = build_style_dataset(
            d, d_filters, d_filter_config, data_config=stylecf_data_config
        )

        batch_x, batch_y = next(iter(train_loader))

        self.assertIn("enc_x", batch_x.keys())
        self.assertIn("dec_x", batch_x.keys())
        self.assertIn("style", batch_x.keys())
        self.assertIn("y_seq", batch_y.keys())

        for key in ("enc_x", "dec_x", "style", "y_seq"):
            tensor = batch_x[key] if key in batch_x.keys() else batch_y[key]
            self.assertIsNotNone(tensor.names)
            self.assertEqual(tensor.names[-2:], (TensorNames.T, TensorNames.F))
            print(tensor.shape)
    
