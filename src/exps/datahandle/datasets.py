from __future__ import annotations

from typing import Callable

import numpy as np
import torch
from tensordict import TensorDict
from torch.utils.data import DataLoader, Dataset

from ...schema import TensorNames
from ...utils.logger import get_with_warn
from ..utils.utils import SliceableTensorDict


def _fit_scaler(scaler, data: np.ndarray):
    shape = data.shape
    flat = data.reshape(-1, shape[-1])
    scaler.fit(flat)
    return scaler


def _transform(scaler, data: np.ndarray):
    shape = data.shape
    flat = data.reshape(-1, shape[-1])
    flat_scaled = scaler.transform(flat)
    return flat_scaled.reshape(shape)



def make_transform(scalers, x_groups):
    scaler_by_key = {}
    if isinstance(scalers, dict):
        for key, group in x_groups.items():
            if group.get("transform", True) and key in scalers:
                scaler_by_key[key] = scalers[key]
    else:
        for idx, (key, group) in enumerate(x_groups.items()):
            if group.get("transform", True):
                scaler_by_key[key] = scalers[idx]

    def _apply_transform(x_payload):
        out = dict(x_payload)
        for key, scaler in scaler_by_key.items():
            if key in out and out[key] is not None:
                out[key] = _transform(scaler, out[key])
        return out

    return _apply_transform



def _to_named_tensor(value, names) -> torch.Tensor:
    if value is None:
        raise ValueError("Cannot convert None to named tensor")
    tensor = value if isinstance(value, torch.Tensor) else torch.tensor(value)
    if tensor.names is None or any(name is None for name in tensor.names):
        tensor = tensor.refine_names(*names)
    return tensor


###### Transformer Dataset ######


class TransformerDataset(Dataset):
    def __init__(
        self,
        x_seq_enc: np.ndarray,
        x_seq_dec: np.ndarray | None,
        x_static: np.ndarray | None,
        y_seq: np.ndarray | None,
        y_static: np.ndarray | None,
        data_config: dict,
        transform: Callable[[dict[str, np.ndarray]], dict[str, np.ndarray]],
    ):

        if transform is not None:
            x_payload = {"enc_x": x_seq_enc, "dec_x": x_seq_dec, "x_static": x_static}
            x_payload = transform(x_payload)
            x_seq_enc = get_with_warn(x_payload, "enc_x", x_seq_enc)
            x_seq_dec = get_with_warn(x_payload, "dec_x", x_seq_dec)
            x_static = get_with_warn(x_payload, "x_static", x_static)

        self.x_seq_enc = _to_named_tensor(x_seq_enc, [TensorNames.N, TensorNames.T, TensorNames.F]).float()
        self.x_seq_dec = _to_named_tensor(x_seq_dec, [TensorNames.N, TensorNames.T, TensorNames.F]).float()
        self.x_static = _to_named_tensor(x_static, [TensorNames.N, TensorNames.F]).float() if x_static is not None else None
        self.y_seq = _to_named_tensor(y_seq, [TensorNames.N, TensorNames.T, TensorNames.F]).float()
        self.y_static = _to_named_tensor(y_static, [TensorNames.N, TensorNames.F]).float() if y_static is not None else None

        self.num_samples = self.x_seq_enc.shape[0]
        self.total_len = self.x_seq_enc.shape[1]

        self.seq_len = data_config["seq_len"]
        self.label_len = data_config["label_len"]
        self.pred_len = data_config["pred_len"]
        self.stride = data_config.get("stride", 1)

        self.indices = [
            (i, t)
            for i in range(self.num_samples)
            for t in range(0, self.total_len - (self.seq_len + self.pred_len) + 1, self.stride)
        ]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i, t = self.indices[idx]

        x_enc = self.x_seq_enc[i, t : t + self.seq_len, :]

        window = self.x_seq_dec[i, t + self.seq_len - self.label_len : t + self.seq_len + self.pred_len, :]
        label_part = window[: self.label_len]
        pred_part = window[self.label_len :].clone()
        pred_part[:, 0] = label_part[:, 0].mean()
        x_dec = torch.cat([label_part, pred_part], dim=0)

        x_static = self.x_static[i] if self.x_static is not None else None
        y_seq = self.y_seq[i, t + self.seq_len - 1 : t + self.seq_len + self.pred_len, :]
        y_static = self.y_static[i] if self.y_static is not None else None

        return (x_enc, x_dec, x_static), (y_seq, y_static)


###### StyleCF Transformer Dataset ######


class StyledTransfollowerDataset(TransformerDataset):
    def __init__(
        self,
        x_seq_enc,
        x_seq_dec,
        x_style,
        y_seq,
        data_config,
        transform,
        sample_self_ids: np.ndarray | None = None,
    ):
        if transform is not None:
            x_payload = {"enc_x": x_seq_enc, "dec_x": x_seq_dec, "style": x_style}
            x_payload = transform(x_payload)
            x_seq_enc = get_with_warn(x_payload, "enc_x", x_seq_enc)
            x_seq_dec = get_with_warn(x_payload, "dec_x", x_seq_dec)
            x_style = get_with_warn(x_payload, "style", x_style)

        super().__init__(x_seq_enc, x_seq_dec, None, y_seq, None, data_config, transform=None)
        self.x_style = _to_named_tensor(x_style, [TensorNames.N, TensorNames.T, TensorNames.F]).float()

        if sample_self_ids is None:
            self.sample_self_ids = None
        else:
            if len(sample_self_ids) != self.num_samples:
                raise ValueError(
                    "sample_self_ids length mismatch: "
                    f"{len(sample_self_ids)} != {self.num_samples}"
                )
            self.sample_self_ids = torch.tensor(sample_self_ids, dtype=torch.long)

        self.style_window_mode = str(data_config.get("style_window_mode", "before_pred_start")).lower()
        self.strict_style_window = bool(data_config.get("strict_style_window"))
        self.sample_dt = float(data_config.get("sample_dt"))
        supported_modes = {"before_pred_start", "before_hist_start"}
        if self.style_window_mode not in supported_modes:
            raise ValueError(
                f"Unsupported style_window_mode: {self.style_window_mode}. "
                "Expected one of: before_pred_start, before_hist_start."
            )

        window_secs = data_config.get("style_window_before_seconds")
        if isinstance(window_secs, (list, tuple)) and len(window_secs) == 2:
            near_s, far_s = float(window_secs[0]), float(window_secs[1])
        else:
            raise ValueError("Expected a list or tuple of two floats: [near_seconds, far_seconds].")
        near_s, far_s = min(near_s, far_s), max(near_s, far_s)
        self.style_near_steps = max(1, int(round(near_s / self.sample_dt)))
        self.style_far_steps = max(self.style_near_steps + 1, int(round(far_s / self.sample_dt)))

        if self.strict_style_window:
            self.indices = [
                (i, t)
                for (i, t) in self.indices
                if self._style_window_bounds(t)[0] >= 0
            ]
            if not self.indices:
                raise ValueError(
                    "No valid windows after applying strict style window "
                    f"[{near_s}, {far_s}] with style_window_mode='{self.style_window_mode}'."
                )

    def _style_window_bounds(self, t: int) -> tuple[int, int]:
        if self.style_window_mode == "before_pred_start":
            anchor = t + self.seq_len
        else:
            anchor = t
        start = anchor - self.style_far_steps
        end = anchor - self.style_near_steps
        return start, end

    def _slice_style(self, i: int, t: int) -> torch.Tensor:
        start, end = self._style_window_bounds(t)
        if start < 0 or end <= start:
            if self.strict_style_window:
                raise ValueError(
                    f"Invalid style window bounds start={start}, end={end}. "
                    "Disable strict_style_window to clamp boundaries."
                )
            end = max(1, min(end, self.total_len))
            length = max(1, self.style_far_steps - self.style_near_steps)
            start = max(0, end - length)

        end = min(end, self.total_len)
        return self.x_style[i, start:end, :]

    def __getitem__(self, idx):
        (x_enc, x_dec, x_static), (y_seq, y_static) = super().__getitem__(idx)
        i, t = self.indices[idx]

        x_style = self._slice_style(i, t)

        x = TensorDict({"enc_x": x_enc, "dec_x": x_dec, "style": x_style}, batch_size=[])
        y_payload = {"y_seq": y_seq}
        if self.sample_self_ids is not None:
            y_payload["self_id"] = self.sample_self_ids[i : i + 1].refine_names(TensorNames.F)
        y = TensorDict(y_payload, batch_size=[])
        return x, y


###### LSTM Dataset ######


class LSTMDataset(Dataset):
    def __init__(self, micro_x, micro_y, data_config):
        assert micro_x.shape[1] == micro_y.shape[1], "micro_x and micro_y must have the same time length"

        self.train_step = data_config["seq_len"]
        self.pred_step = data_config["pred_len"]
        self.stride = data_config.get("stride", 1)

        self.micro_x = torch.tensor(micro_x, dtype=torch.float32)
        self.micro_y = torch.tensor(micro_y, dtype=torch.float32)

        self.total_steps = micro_x.shape[1]
        self.num_samples = micro_x.shape[0]

        self.seq_per_sample = (
            (self.total_steps - self.train_step - self.pred_step) // self.stride + 1
        )
        self.total_seq = self.num_samples * self.seq_per_sample

    def __len__(self):
        return self.total_seq

    def __getitem__(self, idx):
        batch_idx = idx // self.seq_per_sample
        offset = idx % self.seq_per_sample
        time_idx = offset * self.stride

        x_seq = self.micro_x[batch_idx, time_idx : time_idx + self.train_step, :]
        y_seq = self.micro_y[
            batch_idx,
            time_idx + self.train_step - 1 : time_idx + self.train_step + self.pred_step,
            :,
        ]

        return x_seq, y_seq


###### IDM Dataset ###


class IDMDataset(Dataset):
    def __init__(self, x: np.ndarray, y_self: np.ndarray, y_leader: np.ndarray, downsample_step: int = 1):
        self.x = _to_named_tensor(x, [TensorNames.N, TensorNames.T, TensorNames.F]).float()
        self.y_self = y_self
        self.y_leader = y_leader
        self.downsample_step = downsample_step

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        x_t = self.x[idx, :: self.downsample_step, :]
        y_self_t = self.y_self[idx, :: self.downsample_step, :]
        y_leader_t = self.y_leader[idx, :: self.downsample_step, :]

        x_td = SliceableTensorDict({TensorNames.INPUTS: x_t}, batch_size=[])
        y_td = SliceableTensorDict({"self_move": y_self_t, "leader_move": y_leader_t}, batch_size=[])
        return x_td, y_td
