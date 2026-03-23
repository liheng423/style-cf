# ========= Transformer Model =========== #



import torch
import torch.nn as nn
from ..agent import Agent
from ..utils.utils import SliceableTensorDict, restore_tensor_names_like, strip_tensor_names
from ..utils.utils_namebuilder import _build_name_dict
from ...schema import CFNAMES as CF
from typing import Any, Callable, Mapping, Protocol, Sequence, Tuple, Union, cast, runtime_checkable


class Scaler(Protocol):
    def transform(self, x: Any) -> Any: ...
    def inverse_transform(self, x: Any) -> Any: ...


@runtime_checkable
class TrainSeriesWithInputs(Protocol):
    inputs: Tuple[Any, ...]


def _resolve_enc_dec_scalers(simulator: Agent) -> tuple[Scaler, Scaler]:
    if hasattr(simulator, "scalers"):
        scalers = cast(Union[Mapping[str, Scaler], Sequence[Scaler]], simulator.scalers)
    elif hasattr(simulator, "scaler"):
        scalers = cast(Union[Mapping[str, Scaler], Sequence[Scaler]], simulator.scaler)
    else:
        raise AttributeError("Simulator has no scalers or scaler attribute")

    if isinstance(scalers, Mapping):
        enc_scaler = scalers.get("enc_x")
        dec_scaler = scalers.get("dec_x")
        if enc_scaler is None or dec_scaler is None:
            raise KeyError("scalers must include 'enc_x' and 'dec_x'")
        return enc_scaler, dec_scaler

    if not isinstance(scalers, Sequence):
        raise TypeError("scalers must be a mapping or a sequence")

    if len(scalers) < 2:
        raise ValueError("scalers sequence must contain at least two elements")

    return scalers[0], scalers[1]


TrainSeries = Union[SliceableTensorDict, TrainSeriesWithInputs]


def build_causal_tgt_mask(
    tgt_len: int,
    device: torch.device,
    unmask_first_col: bool = False,
) -> torch.Tensor:
    """
    Build an additive causal attention mask for decoder self-attention.

    Values:
    - 0.0: can attend
    - -inf: blocked
    """
    if tgt_len <= 0:
        raise ValueError("tgt_len must be positive")
    tgt_mask = torch.triu(
        torch.full((tgt_len, tgt_len), float("-inf"), device=device),
        diagonal=1,
    )
    if unmask_first_col:
        tgt_mask[:, 0] = 0.0
    return tgt_mask


def transformer_lead_update_func(simulator: Agent, data_config: dict) -> Callable[[TrainSeries, torch.Tensor], TrainSeries]:
    """
    Update decoder inputs when only leader movements are known.
    """
    x_groups = data_config.get("x_groups", data_config)
    name_dict = _build_name_dict(x_groups)
    leader_v_idx = name_dict["dec_x"][CF.LEAD_V]
    enc_scaler, dec_scaler = _resolve_enc_dec_scalers(simulator)

    def _update_train_series_lead(train_series: TrainSeries, leader_movements: torch.Tensor) -> TrainSeries:
        if hasattr(train_series, "inputs"):
            train_with_inputs = cast(TrainSeriesWithInputs, train_series)
            enc_series = train_with_inputs.inputs[0]
            dec_series = train_with_inputs.inputs[1]
            rest = train_with_inputs.inputs[2:]

            enc_series = enc_scaler.inverse_transform(enc_series)
            dec_series = dec_scaler.inverse_transform(dec_series)

            dec_series[:, leader_v_idx] = leader_movements[:, 1]

            enc_scaled = torch.tensor(enc_scaler.transform(enc_series)).float()
            dec_scaled = torch.tensor(dec_scaler.transform(dec_series)).float()

            train_with_inputs.inputs = (enc_scaled, dec_scaled, *rest)
            return train_with_inputs

        if isinstance(train_series, SliceableTensorDict):
            dec_series = train_series["dec_x"]
            dec_names = dec_series.names
            dec_series = dec_scaler.inverse_transform(dec_series)
            dec_series[:, leader_v_idx] = leader_movements[:, 1]
            dec_scaled = torch.tensor(dec_scaler.transform(dec_series)).float()
            if dec_names is not None:
                dec_scaled = dec_scaled.refine_names(*dec_names)

            out = {key: train_series[key] for key in train_series.keys()}
            out["dec_x"] = dec_scaled
            td_cls = type(train_series)
            return td_cls(out, batch_size=train_series.batch_size, names=train_series.names)

        raise TypeError("Unsupported train_series type for leader update")

    return _update_train_series_lead



def transformer_update_func(simulator: Agent, data_config: dict):
    """
    Update encoder/decoder inputs for transformer-style models (no style token).
    """

    name_dict = _build_name_dict(data_config["x_groups"])
    simulator._update_train_series_lead = transformer_lead_update_func(simulator, data_config)

    def _update_train_series(train_series: SliceableTensorDict, self_movements: torch.Tensor, leader_movements: torch.Tensor):
        """
        Args:
            train_series: SliceableTensorDict with keys "enc_x" and "dec_x".
            self_movements: torch.Tensor (time, [x_self, v_self, a_self])
            leader_movements: torch.Tensor (time, [x_self, v_self, a_self])
        """
        enc_series = train_series["enc_x"]
        dec_series = train_series["dec_x"]
        enc_names = enc_series.names
        dec_names = dec_series.names

        enc_series = simulator.scalers["enc_x"].inverse_transform(enc_series)
        dec_series = simulator.scalers["dec_x"].inverse_transform(dec_series)

        delta_x = leader_movements[:, 0] - self_movements[:, 0]
        delta_v = leader_movements[:, 1] - self_movements[:, 1]

        # Update encoder series: [v_self, delta_x, delta_v]
        enc_series[:, name_dict["enc_x"][CF.SELF_V]] = self_movements[:, 1]
        enc_series[:, name_dict["enc_x"][CF.DELTA_X]] = delta_x
        enc_series[:, name_dict["enc_x"][CF.DELTA_V]] = delta_v

        # Update decoder series: [v_self, v_leader]
        dec_series[:, name_dict["dec_x"][CF.SELF_V]] = self_movements[:, 1]
        dec_series[:, name_dict["dec_x"][CF.LEAD_V]] = leader_movements[:, 1]

        # Repack and rescale
        enc_series_scaled = torch.tensor(simulator.scalers["enc_x"].transform(enc_series)).float()
        dec_series_scaled = torch.tensor(simulator.scalers["dec_x"].transform(dec_series)).float()
        if enc_names is not None:
            enc_series_scaled = enc_series_scaled.refine_names(*enc_names)
        if dec_names is not None:
            dec_series_scaled = dec_series_scaled.refine_names(*dec_names)

        out = {"enc_x": enc_series_scaled, "dec_x": dec_series_scaled}
        td_cls = type(train_series)
        return td_cls(out, batch_size=train_series.batch_size, names=train_series.names)

    return _update_train_series



def transformer_mask(data_config: dict):
    """
    Construct a masked decoder input by combining past (from seq_data) and future (from pred_data),
    and masking the future portion to avoid leakage.
    """
    name_dict = _build_name_dict(data_config["x_groups"])

    def _mask(seq_data: SliceableTensorDict, pred_data: SliceableTensorDict, *args) -> SliceableTensorDict:
        label_len = data_config["label_len"]
        pred_len = data_config["pred_len"]

        # encoder input stays the same
        enc_seq_series = cast(torch.Tensor, seq_data["enc_x"])  # shape: (seq_len, dim)
        dec_seq_series = cast(torch.Tensor, seq_data["dec_x"])
        dec_pred_series = cast(torch.Tensor, pred_data["dec_x"])

        # take last label_len rows from seq_data's decoder input
        dec_past = strip_tensor_names(dec_seq_series[-label_len:].clone())  # shape: (label_len, dim)
        dec_pred_series = strip_tensor_names(dec_pred_series)

        dec_future = dec_past.new_zeros((pred_len, dec_past.shape[-1]))

        # take pred_len rows from pred_data's decoder input
        leader_v_idx = name_dict["dec_x"][CF.LEAD_V]
        self_v_idx = name_dict["dec_x"][CF.SELF_V]
        leader_v_pred = dec_pred_series[:, leader_v_idx].clone()  # shape: (pred_len, dim)

        # compute the mean of feature 0 in the past (label_len) decoder steps
        mean_val = torch.mean(dec_past[:, self_v_idx])

        # mask feature 0 in future decoder input
        dec_future[:, self_v_idx] = mean_val
        dec_future[:, leader_v_idx] = leader_v_pred

        

        # concatenate past and masked future
        dec_series_masked = torch.cat([dec_past, dec_future], dim=0)  # shape: (label_len + pred_len, dim)
        dec_series_masked = restore_tensor_names_like(dec_series_masked, dec_seq_series)

        out = {"enc_x": enc_seq_series, "dec_x": dec_series_masked}
        return SliceableTensorDict(out, batch_size=seq_data.batch_size, names=seq_data.names)


    return _mask

class Transfollower(nn.Module):
    def __init__(self, transfollower_config, d_model = 256, num_encoder_layers = 2, num_decoder_layers = 1):
        super(Transfollower, self).__init__()
        enc_in, dec_in = transfollower_config["enc_in"], transfollower_config["dec_in"]
        self.transformer = nn.Transformer(d_model= d_model, nhead=8, num_encoder_layers=num_encoder_layers,
                                   num_decoder_layers=num_decoder_layers, dim_feedforward=1024, 
                                   dropout=0, activation='relu', custom_encoder=None,
                                   custom_decoder=None, layer_norm_eps=1e-05, batch_first=True, 
                                   device=None, dtype=None)
        self.enc_emb = nn.Linear(enc_in, d_model)
        self.dec_emb = nn.Linear(dec_in, d_model)
        self.out_proj = nn.Linear(d_model, 1, bias = True)
        self.settings = transfollower_config
        
        self.enc_positional_embedding = nn.Embedding(self.settings["seq_len"], d_model)
        self.dec_positional_embedding = nn.Embedding(self.settings["pred_len"] + self.settings["label_len"], d_model)

        nn.init.normal_(self.enc_emb.weight, 0, .02)
        nn.init.normal_(self.dec_emb.weight, 0, .02)
        nn.init.normal_(self.out_proj.weight, 0, .02)
        nn.init.normal_(self.enc_positional_embedding.weight, 0, .02)
        nn.init.normal_(self.dec_positional_embedding.weight, 0, .02)

    def forward(self, x: SliceableTensorDict):
        enc_inp, dec_inp = x["enc_x"], x["dec_x"] # (Batch, time, feature)
        enc_pos = torch.arange(0, enc_inp.shape[1]).to(enc_inp.device)
        dec_pos = torch.arange(0, dec_inp.shape[1]).to(dec_inp.device)
        enc_inp = self.enc_emb(enc_inp) + self.enc_positional_embedding(enc_pos)[None,:,:]
        dec_inp = self.dec_emb(dec_inp) + self.dec_positional_embedding(dec_pos)[None,:,:]

        tgt_mask = build_causal_tgt_mask(dec_inp.shape[1], dec_inp.device)
        transformer_out = self.transformer(enc_inp, dec_inp, tgt_mask=tgt_mask) # out: (Batch, time, feature)
        out = self.out_proj(transformer_out)
        return out[:,-self.settings["pred_len"]:,:].squeeze(2)
    
