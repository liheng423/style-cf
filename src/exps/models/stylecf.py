from typing import List, cast
from tensordict import TensorDict
from typing import Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from exps.models.transfollower import transformer_update_func, transformer_mask
from src.exps.agent import Agent
from src.exps.utils.utils import SliceableTensorDict, drop_tensor_names
from src.schema import CFNAMES as CF
from typing import Protocol


class StyleModel(Protocol):
    
    use_dummy_style: bool
    
    def forward(self, x: TensorDict) -> tuple[torch.Tensor, torch.Tensor]:
        ...
        
    def to(self, device: torch.device) -> None:
        ...
        
    def load_state_dict(self, state_dict: dict) -> None:
        ...
        
    

#
#class LossFunction:
#
#    @staticmethod
#    def _predict_kinematics(accs: torch.Tensor, ground_truth: torch.Tensor, dt: float):
#        """
#        Compute predicted speed and distance based on acceleration.
#        
#        Args:
#            accs (torch.Tensor): Acceleration tensor of shape (N, T).
#            ground_truth (torch.Tensor): Ground truth tensor of shape (N, T+1, [distance, speed, acc]).
#            dt (float): Time step duration.
#        
#        Returns:
#            pred_spd (torch.Tensor): Predicted speed tensor of shape (N, T).
#            pred_dis (torch.Tensor): Predicted distance tensor of shape (N, T).
#        """
#        init_dis, init_spd = ground_truth[:, 0, 0].unsqueeze(1), ground_truth[:, 0, 1].unsqueeze(1)
#        pred_spd = init_spd + torch.cumsum(accs, dim=1) * dt
#        # pred_dis = init_dis + torch.cumsum(pred_spd * dt + 0.5 * accs * dt**2, dim=1)
#        pred_dis = init_dis + torch.cumsum(pred_spd , dim=1) * dt
#        return pred_spd, pred_dis
#    
#    @staticmethod
#    def acc_spacing_mse(accs: torch.Tensor, ground_truth: torch.Tensor, dt: float):
#        _, pred_dis = LossFunction._predict_kinematics(accs, ground_truth, dt)
#        true_leader_dis = ground_truth[:, 1:, 4]
#        true_spacing = ground_truth[:, 1:, 3]
#        return F.mse_loss(true_leader_dis - pred_dis, true_spacing)
#    
#class StyleCFLoss:
#
#    @staticmethod
#    def acc_spacing_mse(outputs, y, dt):
#        output_accs, output_style = outputs
#        y_traj = y
#        acc_loss = LossFunction.acc_spacing_mse(output_accs, y_traj, dt)
#        return acc_loss    
#
#
class TransfollowerStyleToDecoder(nn.Module):
    """
    Transformer model that incorporates a style embedding token into both the encoder and decoder inputs.
    """
    def __init__(self, transfollower_config, d_model=256, num_encoder_layers=1, num_decoder_layers=1):
        super(TransfollowerStyleToDecoder, self).__init__()
        self.d_model = d_model
        self.settings = transfollower_config

        enc_in, dec_in = len(self.settings["x_groups"]["enc_x"]["features"]), len(self.settings["x_groups"]["dec_x"]["features"])

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=8,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=1024,
            dropout=0,
            activation='relu',
            batch_first=True,
        )

        # Embedding layers
        self.enc_emb = nn.Linear(enc_in, d_model)
        self.dec_emb = nn.Linear(dec_in, d_model)
        self.style_proj = nn.Linear(d_model, d_model)  # Projection for style embedding

        self.out_proj = nn.Linear(d_model, 1)

        # Positional embeddings
        self.positional_embedding = nn.Embedding(self.settings["seq_len"] + self.settings["pred_len"], d_model)

        # Initialization
        nn.init.normal_(self.enc_emb.weight, 0, .02)
        nn.init.normal_(self.dec_emb.weight, 0, .02)
        nn.init.normal_(self.out_proj.weight, 0, .02)
        nn.init.normal_(self.positional_embedding.weight, 0, .02)
        nn.init.normal_(self.style_proj.weight, 0, .02)

    def forward(self, x):
        """
        x: tuple(enc_inp, dec_inp, d_style)
        - enc_inp: (B, T_enc, enc_in)
        - dec_inp: (B, T_dec, dec_in)
        - d_style: (B, d_model)
        """
        enc_inp, dec_inp, d_style = x
        B, T_enc = enc_inp.shape[0], enc_inp.shape[1]
        T_dec = dec_inp.shape[1]

        device = enc_inp.device

        # === Style token ===
        style_token = self.style_proj(d_style).unsqueeze(1)  # (B, 1, D), no positional encoding added

        # === Encoder input ===
        enc_pos = torch.arange(0, T_enc).to(device)
        enc_embed = self.enc_emb(enc_inp) + self.positional_embedding(enc_pos)[None, :, :]  # (B, T_enc, D)

        enc_embed = torch.cat([style_token, enc_embed], dim=1)

        # === Decoder input ===
        dec_pos = torch.arange(T_enc - self.settings["label_len"], T_enc + self.settings["pred_len"]).to(device)
        dec_embed = self.dec_emb(dec_inp) + self.positional_embedding(dec_pos)[None, :, :]  # (B, T_dec, D)

        # Concatenate style token at the beginning of the decoder sequence
        dec_embed = torch.cat([style_token, dec_embed], dim=1)  # (B, T_dec+1, D)
        # dec_embed = style_token + dec_embed

        # === Transformer forward ===
        transformer_out = self.transformer(
            src=enc_embed,
            tgt=dec_embed,
            tgt_mask=self.transformer.generate_square_subsequent_mask(T_dec).to(device),
            tgt_is_causal=True,
        )  # (B, T_dec+1, D)

        # Output projection
        out = self.out_proj(transformer_out)  # (B, T_dec+1, 1)

        # Only take the predictions for the last `pred_len` time steps (excluding the style token)
        return out[:, -self.settings["pred_len"]:, :].squeeze(2)  # (B, pred_len)

class StyleEmbedder(nn.Module):
    """
    Style embedding module using a Transformer encoder.
    """
    def __init__(self, input_dim, embed_dim=256, num_heads=8, num_layers=2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Use Linear to aggregate the time dimension (e.g., T time steps -> 1 embedding)
        self.time_fc = nn.Linear(embed_dim, 1)  # Score each time step
        self.output_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        """
        x: (B, T, input_dim)
        """
        x = self.input_proj(x)         # (B, T, E)
        x = x.permute(1, 0, 2)         # (T, B, E) for transformer
        x = self.transformer(x)        # (T, B, E)
        x = x.permute(1, 0, 2)         # (B, T, E)

        # Time-weighted average: Use a Linear layer to calculate attention weights (similar to Attention Pooling)
        attn_weights = torch.softmax(self.time_fc(x).squeeze(-1), dim=1)  # (B, T)
        x = (x * attn_weights.unsqueeze(-1)).sum(dim=1)  # Weighted sum, (B, E)

        x = self.output_proj(x)        # Optional further mapping
        return F.normalize(x, p=2, dim=1)



class StyleTransformer(nn.Module, StyleModel):
    """
    Transformer model that integrates style embeddings into car-following predictions.
    """
    def __init__(self, transfollower_config, embed_dim=256, num_heads=8, num_enc_layers=1, num_dec_layers=1):
        super().__init__()
        input_dim = 4
        self.embedder = StyleEmbedder(input_dim, embed_dim, num_heads, num_enc_layers)
        self.transfollower = TransfollowerStyleToDecoder(transfollower_config, d_model=embed_dim,
                                                num_encoder_layers=num_enc_layers,
                                                num_decoder_layers=num_dec_layers)
        self.use_dummy_style = False
        self.embed_dim = embed_dim

    def forward(self, x: TensorDict):
        """
        x: TensorDict with keys "enc_x", "dec_x", "style"
        - enc_x: (B, T_enc, enc_in)
        - dec_x: (B, T_dec, dec_in)
        - style: (B, T_style, d_style)
        """
        enc_inp = drop_tensor_names(cast(torch.Tensor, x["enc_x"]))
        dec_inp = drop_tensor_names(cast(torch.Tensor, x["dec_x"]))
        style = drop_tensor_names(cast(torch.Tensor, x["style"]))
        B = enc_inp.size(0)

        if self.use_dummy_style:
            d_style = torch.randn(B, self.embed_dim, device=enc_inp.device)  # Dummy random vector
        else:
            d_style = self.embedder(style)  # Use real style embedding

        accs = self.transfollower((enc_inp, dec_inp, d_style))
        return accs, d_style

######## MASK ##########
def stylecf_mask(data_config):
    """
    Construct a masked decoder input by combining past (from seq_data) and future (from pred_data),
    and masking the future portion to avoid leakage.
    """
    base_mask = transformer_mask(data_config)

    def _mask(seq_data: SliceableTensorDict, pred_data: SliceableTensorDict, *_: Any) -> SliceableTensorDict:
        out = base_mask(seq_data, pred_data)
        style = cast(torch.Tensor, seq_data["style"])
        assert not torch.isnan(style).any()
        out["style"] = style
        return out
    
    return _mask





def style_update_func(simulator: Agent, featuer_dict: dict[str, List[str]]): 
    update_transformer = transformer_update_func(simulator, featuer_dict)

    def _update_train_series(train_series: SliceableTensorDict, self_movements: torch.Tensor, leader_movements: torch.Tensor):
        """

        Style is not updated, since each style token is determined before testing.

        Args:
            train_series: SliceableTensorDict 
            self_movements : torch.Tensor  (time, [x_self, v_self, a_self])
            leader_movements : torch.Tensor (time, [x_self, v_self, a_self])
        """

        out = update_transformer(train_series, self_movements, leader_movements)
        if "style" in train_series.keys():
            style = train_series["style"]
            if style is not None:
                out["style"] = style
        return out
    
    return _update_train_series
