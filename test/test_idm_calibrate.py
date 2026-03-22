import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.idm_calibrate.calibrator import calibrate_idm
from src.schema import CFNAMES as CF


class FakeSampleDataPack:
    def __init__(self, data, names):
        self.data = data
        self.names = names

    def __getitem__(self, key):
        if isinstance(key, tuple) and len(key) == 3:
            i, j, feat = key
            if isinstance(feat, str):
                return self.data[i, j, self.names[feat]]
            if isinstance(feat, list):
                indices = [self.names[f] for f in feat]
                base = self.data[i, j, :]
                return np.take(base, indices, axis=-1)
        return self.data[key]


def _build_sample_datapack(self_id: int) -> FakeSampleDataPack:
    names = {
        CF.SELF_ID: 0,
        CF.SELF_X: 1,
        CF.SELF_V: 2,
        CF.SELF_A: 3,
        CF.SELF_L: 4,
        CF.LEAD_ID: 5,
        CF.LEAD_X: 6,
        CF.LEAD_V: 7,
        CF.LEAD_A: 8,
        CF.LEAD_L: 9,
        CF.DELTA_X: 10,
        CF.DELTA_V: 11,
    }

    t = np.arange(10, dtype=np.float32)
    self_v = np.full((10,), 10.0, dtype=np.float32)
    lead_v = np.full((10,), 12.0, dtype=np.float32)
    self_x = self_v * t * 0.1
    lead_x = self_x + 20.0
    self_a = np.zeros((10,), dtype=np.float32)
    lead_a = np.zeros((10,), dtype=np.float32)

    data = np.zeros((1, 10, len(names)), dtype=np.float32)
    data[0, :, names[CF.SELF_ID]] = float(self_id)
    data[0, :, names[CF.SELF_X]] = self_x
    data[0, :, names[CF.SELF_V]] = self_v
    data[0, :, names[CF.SELF_A]] = self_a
    data[0, :, names[CF.LEAD_ID]] = float(self_id + 1000)
    data[0, :, names[CF.LEAD_X]] = lead_x
    data[0, :, names[CF.LEAD_V]] = lead_v
    data[0, :, names[CF.LEAD_A]] = lead_a
    data[0, :, names[CF.DELTA_X]] = lead_x - self_x
    data[0, :, names[CF.DELTA_V]] = lead_v - self_v

    return FakeSampleDataPack(data, names)


class TestIdmCalibrate(unittest.TestCase):
    def test_calibrate_idm_clamps_sample_size_and_saves_csv(self):
        id_datapack = {
            0: _build_sample_datapack(101),
            1: _build_sample_datapack(102),
            2: _build_sample_datapack(103),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            class DummyIDMDataset:
                def __init__(self, x, y_self, y_leader, downsample_step):
                    self.item = (x, {"self_move": y_self, "leader_move": y_leader})

                def __len__(self):
                    return 1

                def __getitem__(self, idx):
                    return self.item

            def dummy_dataloader(dataset, *args, **kwargs):
                return [dataset[0]]

            config = {
                "x_groups": {"x": {"features": [CF.SELF_V, CF.LEAD_V, CF.DELTA_X]}},
                "y_groups": {"y": {"features": [CF.SELF_X, CF.SELF_V, CF.SELF_A]}},
                "downsample": 1,
                "save_path": tmpdir,
                "device": "cpu",
                "sample_size": 1000,
                "randomseed": 7,
                "dataset_cls": DummyIDMDataset,
                "dataloader_cls": dummy_dataloader,
            }

            with patch("src.idm_calibrate.calibrator.calibrate_idm_genetic") as mocked_genetic:
                mocked_genetic.return_value = (np.array([24.0, 1.0, 1.2, 1.4, 2.0]), np.array([0.123]))
                df = calibrate_idm(object, id_datapack, config)

            self.assertEqual(mocked_genetic.call_count, 3)
            self.assertEqual(len(df), 3)
            self.assertSetEqual(set(df.columns), {"ID", "v0", "s0", "T", "a", "b", "best_loss"})
            self.assertSetEqual(set(df["ID"].astype(int).tolist()), {101, 102, 103})

            output_path = Path(tmpdir) / "idm_calibration.csv"
            self.assertTrue(output_path.exists())

            loaded = pd.read_csv(output_path)
            self.assertEqual(len(loaded), 3)


if __name__ == "__main__":
    unittest.main()
