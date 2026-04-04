from __future__ import annotations

import numpy as np

from ..datapack import BASE_CF_NAMES, SampleDataPack


def load_zen_data(path, rise, in_kph=False, kilo_norm=False):
    data: np.ndarray = np.load(path, allow_pickle=True).astype(np.float32)
    print(f"Data Shape: {data.shape}")
    return SampleDataPack(data, BASE_CF_NAMES.copy(), rise=rise, kph=in_kph, kilo_norm=kilo_norm, dt=0.1)

