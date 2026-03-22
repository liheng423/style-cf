from __future__ import annotations

from typing import Dict, List

import numpy as np

from .utils_kine import _predict_kinematics_np
from ...schema import CFNAMES as CF


class SampleDataPack:
    """
    Manage and manipulate car-following data stored in a (sample, time, feature) array.
    """

    def __init__(self, data: np.ndarray, name_dict: dict, rise: bool, kph: bool, kilo_norm, dt: float):
        self.data = data
        self.names = name_dict
        self.rise = rise
        self.kph = kph
        self.kilo_norm = kilo_norm
        self.dt = dt

    def append_col(self, col: np.ndarray, col_name: str):
        assert col_name not in self.names
        assert col.shape[:2] == self.data.shape[:2]

        if len(col.shape) == 2:
            col = col[:, :, np.newaxis]

        col_index = self.data.shape[2]
        self.data = np.concatenate([self.data, col], axis=2)
        self.names[col_name] = col_index

    def replace_col(self, col: np.ndarray, col_name: str):
        assert col_name in self.names, f"Column '{col_name}' not found."
        assert col.shape[:2] == self.data.shape[:2], "Shape mismatch with existing data."

        if len(col.shape) == 2:
            col = col[:, :, np.newaxis]

        col_index = self.names[col_name]
        self.data[:, :, col_index] = col[:, :, 0]

    def head(self, n: int) -> "SampleDataPack":
        return SampleDataPack(self.data[:n], self.names.copy(), self.rise, self.kph, self.kilo_norm, self.dt)

    def __getitem__(self, key) -> np.ndarray:
        if isinstance(key, tuple) and len(key) == 3:
            i, j, feat = key
            if isinstance(feat, str):
                k = self.names[feat]
                return self.data[i, j, k]
            if isinstance(feat, list):
                k = [self.names[f] for f in feat]
                base = self.data[i, j, :]
                return np.take(base, k, axis=-1)
            raise TypeError("Third index must be a string or list of strings.")
        return self.data[key]

    def reorder_features(self, new_name_dict: dict):
        if set(new_name_dict.keys()) != set(self.names.keys()):
            raise ValueError("New name_dict must contain the same keys as the original.")

        new_order = [self.names[name] for name in sorted(new_name_dict, key=lambda x: new_name_dict[x])]
        self.data = self.data[:, :, new_order]
        self.names = new_name_dict

    def normalize_kilopost(self, column_keys: List[str] = [CF.LEAD_X, CF.SELF_X]) -> "SampleDataPack":
        if self.kilo_norm is True:
            print("The kilopost is already normalized, nothing is done")
            return self

        column_indices = []
        for key in column_keys:
            idx = self.names.get(key)
            if idx is None:
                raise KeyError(f"Key '{key}' not found in names dict.")
            column_indices.append(idx)

        new_data = self.data.copy()
        extracted = np.stack([new_data[:, :, idx] for idx in column_indices], axis=1)

        mins = np.min(extracted, axis=(1, 2), keepdims=True)
        maxs = np.max(extracted, axis=(1, 2), keepdims=True)
        normalized = extracted - mins if self.rise else maxs - extracted

        for i, idx in enumerate(column_indices):
            new_data[:, :, idx] = normalized[:, i, :]

        return SampleDataPack(new_data, self.names.copy(), kilo_norm=True, kph=self.kph, rise=True, dt=self.dt)

    def split_by_time_windows(self, windows: list[tuple[int, int]]) -> "SampleDataPack":
        for start, end in windows:
            if not (0 <= start < end <= self.data.shape[1]):
                raise ValueError(f"Invalid window ({start}, {end})")

        split_data = np.concatenate([self.data[:, start:end, :].copy() for (start, end) in windows], axis=0)
        return SampleDataPack(split_data, self.names.copy(), self.rise, self.kph, self.kilo_norm, dt=self.dt)

    def split_by_time_windows_list(self, windows: list[tuple[int, int]]) -> list["SampleDataPack"]:
        packs: list[SampleDataPack] = []
        for start, end in windows:
            if not (0 <= start < end <= self.data.shape[1]):
                raise ValueError(f"Invalid window ({start}, {end})")
            split_data = self.data[:, start:end, :].copy()
            packs.append(SampleDataPack(split_data, self.names.copy(), self.rise, self.kph, self.kilo_norm, dt=self.dt))
        return packs

    def convert_speed_to_ms(self, cols: list[str]) -> "SampleDataPack":
        if not self.kph:
            print("The data is already in m/s, nothing is executed.")
            return self

        for col in cols:
            if col not in self.names:
                raise ValueError(f"Column '{col}' not found in data.")

        new_data = self.data.copy()
        for col in cols:
            idx = self.names[col]
            new_data[:, :, idx] = new_data[:, :, idx] / 3.6

        return SampleDataPack(new_data, self.names.copy(), self.rise, False, self.kilo_norm, self.dt)

    def check_consistency(
        self,
        start_idx: int = 1,
        x_key: str = CF.SELF_X,
        v_key: str = CF.SELF_V,
        a_key: str = CF.SELF_A,
    ):
        assert self.kph is False

        accs = self[:, :, a_key][:, start_idx:]
        x = self[:, :, x_key]
        v = self[:, :, v_key]

        initial_states = np.stack([x, v], axis=2)
        initial_states = initial_states[:, start_idx - 1]

        preds = _predict_kinematics_np(accs, initial_states, self.dt)
        pos_error = x[:, start_idx:] - preds[:, :, 0]
        spd_error = v[:, start_idx:] - preds[:, :, 1]
        return pos_error, spd_error

    def force_consistent(self):
        self.replace_col(np.gradient(self[:, :, CF.SELF_X], self.dt, axis=1), CF.SELF_V)
        self.replace_col(np.gradient(self[:, :, CF.SELF_V], self.dt, axis=1), CF.SELF_A)
        self.replace_col(np.gradient(self[:, :, CF.LEAD_X], self.dt, axis=1), CF.LEAD_V)
        self.replace_col(np.gradient(self[:, :, CF.LEAD_V], self.dt, axis=1), CF.LEAD_A)
        self.replace_col(self[:, :, CF.LEAD_V] - self[:, :, CF.SELF_V], CF.DELTA_V)
        self.replace_col(self[:, :, CF.LEAD_X] - self[:, :, CF.SELF_X], CF.DELTA_X)


def load_zen_data(path, rise, in_kph=False, kilo_norm=False):
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
    }

    data: np.ndarray = np.load(path, allow_pickle=True).astype(np.float32)
    print(f"Data Shape: {data.shape}")
    return SampleDataPack(data, names, rise=rise, kph=in_kph, kilo_norm=kilo_norm, dt=0.1)


def build_id_datapack(
    datapack: SampleDataPack,
    require_const_self_id: bool = True,
    key_by_id: bool = False,
) -> Dict[int, SampleDataPack]:
    if CF.SELF_ID not in datapack.names:
        raise KeyError(f"{CF.SELF_ID} not found in datapack.names")

    id_idx = datapack.names[CF.SELF_ID]
    ids = datapack.data[:, :, id_idx]
    first_ids = ids[:, 0]

    if require_const_self_id:
        same_id = np.all(ids == ids[:, [0]], axis=1)
    else:
        same_id = np.ones(ids.shape[0], dtype=bool)

    unique_ids = np.unique(first_ids[same_id])
    id_datapack: Dict[int, SampleDataPack] = {}

    for i, vid in enumerate(unique_ids):
        mask = (first_ids == vid) & same_id
        if not np.any(mask):
            continue
        sub = SampleDataPack(
            datapack.data[mask].copy(),
            datapack.names.copy(),
            datapack.rise,
            datapack.kph,
            datapack.kilo_norm,
            datapack.dt,
        )
        key = int(vid) if key_by_id else i
        id_datapack[key] = sub

    return id_datapack


__all__ = ["SampleDataPack", "build_id_datapack", "load_zen_data"]
