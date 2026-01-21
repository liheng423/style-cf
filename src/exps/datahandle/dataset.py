import torch
from torch.utils.data import Dataset, DataLoader
from tensordict import TensorDict
import numpy as np
from src.stylecf.schema import TensorNames
from typing import Optional
from src.exps.utils.utils import SliceableTensorDict

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


def _to_named_tensor(value, names) -> Optional[torch.Tensor]:
    if value is None:
        return None
    tensor = value if isinstance(value, torch.Tensor) else torch.tensor(value)
    if tensor.names is None or any(name is None for name in tensor.names):
        tensor = tensor.refine_names(*names)
    return tensor

###### Transformer Dataset ######

class TransformerDataset(Dataset):
    def __init__(self, x_seq_enc=None, x_seq_dec=None, x_static=None, y_seq=None, y_static=None, data_config: Optional[dict]=None, transform=None):
        if transform is not None:
            x_payload = {"enc_x": x_seq_enc, "dec_x": x_seq_dec, "x_static": x_static}
            x_payload = transform(x_payload)
            x_seq_enc = x_payload.get("enc_x")
            x_seq_dec = x_payload.get("dec_x")
            x_static = x_payload.get("x_static")

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
        """
        Retrieves a single sample from the dataset.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            tuple: A tuple containing:
                - tuple: (encoder_input, decoder_input, static_features_x)
                - tuple: (sequence_output, static_features_y)
        """
        i, t = self.indices[idx]

        # Encoder input: sequence from time t to t + seq_len
        x_enc = self.x_seq_enc[i, t : t + self.seq_len, :]

        # Decoder input construction
        # The window covers the label_len part (known history) and the pred_len part (to be predicted)
        window = self.x_seq_dec[i, t + self.seq_len - self.label_len : t + self.seq_len + self.pred_len, :]
        label_part = window[:self.label_len]
        pred_part = window[self.label_len:].clone()
        pred_part[:, 0] = label_part[:, 0].mean() # For the first feature in the prediction part, use the mean of the label part
        x_dec = torch.cat([label_part, pred_part], dim=0)

        # Static features and target sequences
        x_static = self.x_static[i] if self.x_static is not None else None
        y_seq = self.y_seq[i, t + self.seq_len - 1 : t + self.seq_len + self.pred_len, :]
        y_static = self.y_static[i] if self.y_static is not None else None

        return (x_enc, x_dec, x_static), (y_seq, y_static)

###### StyleCF Transformer Dataset ######

class StyledTransfollowerDataset(TransformerDataset):
    def __init__(self, x_seq_enc, x_seq_dec, x_style, y_seq, data_config=None, transform=None):
        if transform is not None:
            x_payload = {"enc_x": x_seq_enc, "dec_x": x_seq_dec, "style": x_style}
            x_payload = transform(x_payload)
            x_seq_enc = x_payload.get("enc_x")
            x_seq_dec = x_payload.get("dec_x")
            x_style = x_payload.get("style")
        super().__init__(x_seq_enc, x_seq_dec, None, y_seq, None, data_config)
        self.x_style = _to_named_tensor(x_style, [TensorNames.N, TensorNames.T, TensorNames.F]).float()

    def __getitem__(self, idx):
        (x_enc, x_dec, x_static), (y_seq, y_static) = super().__getitem__(idx)
        i, t = self.indices[idx]

        # 样本 style: 从开头到 t + seq_len 的 window
        x_style = self.x_style[i, t: t + self.seq_len, :]

        x = TensorDict({"enc_x": x_enc, "dec_x": x_dec, "style": x_style}, batch_size=[])
        y = TensorDict({"y_seq": y_seq}, batch_size=[])
        return x, y

###### LSTM Dataset ######

class LSTMDataset(Dataset):
    def __init__(self, micro_x, micro_y, data_config):
        """
        Automatically sliced dataset for LSTM training.

        Args:
            micro_x (np.ndarray or torch.Tensor): (N, total_steps, feature)
            micro_y (np.ndarray or torch.Tensor): (N, total_steps, label_dim)
            train_config (dict): {
                "train_step": int,
                "pred_step": int,
                "stride": int (optional, default=1)
            }
        """
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
        y_seq = self.micro_y[batch_idx, time_idx + self.train_step - 1 : time_idx + self.train_step + self.pred_step, :]

        return x_seq, y_seq


###### IDM Dataset ### 

class IDMDataset(Dataset):
    def __init__(self, x: np.ndarray, y_self: np.ndarray, y_leader: np.ndarray, downsample_step: int = 1):
        """
        Dataset designed for recursive evaluation where two inputs (self_move, leader_move) are needed.

        Args:
            x (np.ndarray): (N, T, [feature])
            y_self (np.ndarray): (N, T, [x, v, a])
            y_leader (np.ndarray): (N, T, [x, v, a])
            downsample_step (int): Step size for downsampling along the time dimension.
        """
        self.x = _to_named_tensor(x, [TensorNames.N, TensorNames.T, TensorNames.F]).float()
        self.y_self = y_self
        self.y_leader = y_leader
        self.downsample_step = downsample_step

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        x_t = self.x[idx, ::self.downsample_step, :]
        y_self_t = self.y_self[idx, ::self.downsample_step, :]
        y_leader_t = self.y_leader[idx, ::self.downsample_step, :]

        x_td = SliceableTensorDict({TensorNames.INPUTS: x_t}, batch_size=[])
        y_td = SliceableTensorDict({"self_move": y_self_t, "leader_move": y_leader_t}, batch_size=[])
        return x_td, y_td
