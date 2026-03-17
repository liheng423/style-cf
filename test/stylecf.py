import os
import unittest
import numpy as np
import test

from src.exps.train.model_trainer import build_style_loader
from src.exps.utils.utils import load_zen_data
from src.exps.configs import style_data_config
from src.schema import CFNAMES as CF
from src.stylecf.schema import TensorNames

from src.exps.train.model_trainer import train_stylecf
from src.exps.models.stylecf import StyleTransformer
from src.exps.configs import *

class TestStyleAgent(unittest.TestCase):

    def _small_dataset(self):

        # data_path = "F:\DATA\ZenTraffic\ZenTraffic30kalman.npy"

        data_path = "/Users/blow/datasets/DATA/ZenTraffic/ZenTraffic30.npy"
        if not os.path.exists(data_path):
            self.skipTest(f"Missing dataset: {data_path}")

        d = load_zen_data(data_path, rise=True, in_kph=False, kilo_norm=True)
        return d.head(300)

    @unittest.skip("Skipping style dataset build test")
    def test_build_style_dataset(self):
        d = self._small_dataset()
        d_filters = [lambda: np.ones(d.data.shape[0], dtype=bool)]
        d_filter_config = {}

        result, _, _, _ = build_style_loader(d, d_filters, d_filter_config, data_config=style_data_config)

        self.assertIn(CF.TIME, result.names)
        self.assertIn(CF.REACT, result.names)
        self.assertIn(CF.THW, result.names)

    @unittest.skip("Skipping style dataloader test")
    def test_style_dataloader_output(self):
        d = self._small_dataset()
        d_filters = [lambda: np.ones(d.data.shape[0], dtype=bool)]
        d_filter_config = {}

        _, train_loader, _, _ = build_style_loader(
            d, d_filters, d_filter_config, data_config=style_data_config
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
            
    def test_style_train(self):
        
        d = self._small_dataset()
        d_filters = [lambda: np.ones(d.data.shape[0], dtype=bool)]
        d_filter_config = {}

        _, train_loader, test_loader, _ = build_style_loader(
            d, d_filters, d_filter_config, data_config=style_data_config
        )


        model = train_stylecf(style_data_config, style_train_config, train_loader, test_loader)
        self.assertIsInstance(model, StyleTransformer)
    
