# ========= Transformer Model =========== #



import torch
import torch.nn as nn
from src.exps.agent import Agent
from src.exps.utils.utils import SliceableTensorDict
from src.exps.utils.utils_namebuilder import _build_name_dict
from src.schema import CFNAMES as CF
from typing import List, cast



def transformer_update_func(simulator: Agent, data_config: dict):
    """
    Update encoder/decoder inputs for transformer-style models (no style token).
    """

    name_dict = _build_name_dict(data_config["x_groups"])

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
        return SliceableTensorDict(out, batch_size=train_series.batch_size, names=train_series.names)

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
        dec_past = dec_seq_series[-label_len:].clone()  # shape: (label_len, dim)

        dec_future = dec_past.new_zeros((pred_len, dec_past.shape[-1]))
        if dec_past.names is not None:
            dec_future = dec_future.refine_names(*dec_past.names)

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
        
        transformer_out = self.transformer(enc_inp, dec_inp) # out: (Batch, time, feature)
        out = self.out_proj(transformer_out)
        return out[:,-self.settings["pred_len"]:,:].squeeze(2)
    