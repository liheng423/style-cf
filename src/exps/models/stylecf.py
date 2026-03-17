from typing import List, cast
from tensordict import TensorDict
from typing import Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from .transfollower import transformer_lead_update_func, transformer_update_func, transformer_mask
from ..agent import Agent
from ..utils.utils import SliceableTensorDict, drop_tensor_names
from ...schema import CFNAMES as CF
from typing import Protocol


class StyleModel(Protocol):
    """
    Minimal interface expected by training/evaluation code for style-aware models.
    """

    use_dummy_style: bool
    
    def forward(self, x: TensorDict) -> tuple[torch.Tensor, torch.Tensor]:
        ...
        
    def to(self, device: torch.device) -> None:
        ...
        
    def load_state_dict(self, state_dict: dict) -> None:
        ...
        
    

class StyleConditionedTransfollower(nn.Module):
    """
    Transformer model that injects a style embedding token into both encoder and decoder streams.

    High-level idea:
    - Encode past observations with a prepended style token.
    - Decode future steps with the same style token prepended to the decoder input.
    - Causal masking ensures the decoder cannot peek into the future.
    """
    def __init__(self, transfollower_config, d_model=256, num_encoder_layers=1, num_decoder_layers=1):
        super(StyleConditionedTransfollower, self).__init__()
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

        # Linear projections from raw feature space -> model dimension.
        self.enc_emb = nn.Linear(enc_in, d_model)
        self.dec_emb = nn.Linear(dec_in, d_model)
        self.style_proj = nn.Linear(d_model, d_model)  # Map style embedding into model dimension

        self.out_proj = nn.Linear(d_model, 1)

        # Positional embeddings for both encoder and decoder time indices.
        self.positional_embedding = nn.Embedding(self.settings["seq_len"] + self.settings["pred_len"], d_model)

        # Initialization
        nn.init.normal_(self.enc_emb.weight, 0, .02)
        nn.init.normal_(self.dec_emb.weight, 0, .02)
        nn.init.normal_(self.out_proj.weight, 0, .02)
        nn.init.normal_(self.positional_embedding.weight, 0, .02)
        nn.init.normal_(self.style_proj.weight, 0, .02)

    def forward(self, x):
        """
        Args:
            x: tuple(enc_inp, dec_inp, d_style)
            - enc_inp: (B, T_enc, enc_in)
            - dec_inp: (B, T_dec, dec_in)
            - d_style: (B, d_model)

        Returns:
            Predicted accelerations with shape (B, pred_len).
        """
        enc_inp, dec_inp, d_style = x
        B, T_enc = enc_inp.shape[0], enc_inp.shape[1]
        T_dec = dec_inp.shape[1]

        device = enc_inp.device

        # === Style token ===
        # The style embedding is a single token that conditions both encoder and decoder.
        style_token = self.style_proj(d_style).unsqueeze(1)  # (B, 1, D), no positional encoding added

        # === Encoder input ===
        # Standard encoder input with positional encoding, plus the prepended style token.
        enc_pos = torch.arange(0, T_enc).to(device)
        enc_embed = self.enc_emb(enc_inp) + self.positional_embedding(enc_pos)[None, :, :]  # (B, T_enc, D)

        enc_embed = torch.cat([style_token, enc_embed], dim=1)

        # === Decoder input ===
        # Decoder input covers label_len past + pred_len future positions with positional encoding.
        dec_pos = torch.arange(T_enc - self.settings["label_len"], T_enc + self.settings["pred_len"]).to(device)
        dec_embed = self.dec_emb(dec_inp) + self.positional_embedding(dec_pos)[None, :, :]  # (B, T_dec, D)

        # Concatenate style token at the beginning of the decoder sequence.
        dec_embed = torch.cat([style_token, dec_embed], dim=1)  # (B, T_dec+1, D)
        # dec_embed = style_token + dec_embed

        # === Transformer forward ===
        # Generate a causal mask for the decoder (length T_dec), while the style token is unmasked.
        transformer_out = self.transformer(
            src=enc_embed,
            tgt=dec_embed,
            tgt_mask=self.transformer.generate_square_subsequent_mask(T_dec).to(device),
            tgt_is_causal=True,
        )  # (B, T_dec+1, D)

        # Output projection
        out = self.out_proj(transformer_out)  # (B, T_dec+1, 1)

        # Only take the predictions for the last `pred_len` time steps (exclude the style token).
        return out[:, -self.settings["pred_len"]:, :].squeeze(2)  # (B, pred_len)

class StyleEmbedder(nn.Module):
    """
    Style embedding module using a Transformer encoder.

    Produces a single L2-normalized vector per sequence using attention-style
    pooling over time.
    """
    def __init__(self, input_dim, embed_dim=256, num_heads=8, num_layers=2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Use a linear layer to score each time step for attention pooling.
        self.time_fc = nn.Linear(embed_dim, 1)
        self.output_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        """
        Args:
            x: (B, T, input_dim)

        Returns:
            Normalized style embedding with shape (B, embed_dim).
        """
        x = self.input_proj(x)         # (B, T, E)
        x = x.permute(1, 0, 2)         # (T, B, E) for transformer
        x = self.transformer(x)        # (T, B, E)
        x = x.permute(1, 0, 2)         # (B, T, E)

        # Time-weighted average: compute attention weights over time steps.
        attn_weights = torch.softmax(self.time_fc(x).squeeze(-1), dim=1)  # (B, T)
        x = (x * attn_weights.unsqueeze(-1)).sum(dim=1)  # Weighted sum, (B, E)

        # Optional projection before normalization.
        x = self.output_proj(x)
        return F.normalize(x, p=2, dim=1)



class StyleTransformer(nn.Module, StyleModel):
    """
    End-to-end model that learns a style embedding and feeds it to the style-aware transformer.
    """
    def __init__(self, transfollower_config, embed_dim=256, num_heads=8, num_enc_layers=1, num_dec_layers=1):
        super().__init__()
        input_dim = 4
        self.embedder = StyleEmbedder(input_dim, embed_dim, num_heads, num_enc_layers)
        self.transfollower = StyleConditionedTransfollower(transfollower_config, d_model=embed_dim,
                                                num_encoder_layers=num_enc_layers,
                                                num_decoder_layers=num_dec_layers)
        self.use_dummy_style = False
        self.embed_dim = embed_dim

    def forward(self, x: TensorDict):
        """
        Args:
            x: TensorDict with keys "enc_x", "dec_x", "style"
            - enc_x: (B, T_enc, enc_in)
            - dec_x: (B, T_dec, dec_in)
            - style: (B, T_style, d_style)

        Returns:
            accs: Predicted accelerations, shape (B, pred_len)
            d_style: Style embedding, shape (B, embed_dim)
        """
        enc_inp = drop_tensor_names(cast(torch.Tensor, x["enc_x"]))
        dec_inp = drop_tensor_names(cast(torch.Tensor, x["dec_x"]))
        style = drop_tensor_names(cast(torch.Tensor, x["style"]))
        B = enc_inp.size(0)

        # Some experiments swap in a random style vector to ablate conditioning.
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
    and masking the future portion to avoid leakage. Style is copied through unmodified.
    """
    base_mask = transformer_mask(data_config)

    def _mask(seq_data: SliceableTensorDict, pred_data: SliceableTensorDict, *_: Any) -> SliceableTensorDict:
        out = base_mask(seq_data, pred_data)
        style = cast(torch.Tensor, seq_data["style"])
        # Style should be fully observed; NaNs indicate missing or invalid style sequences.
        assert not torch.isnan(style).any()
        out["style"] = style
        return out
    
    return _mask





def style_update_func(simulator: Agent, featuer_dict: dict[str, List[str]]): 
    update_transformer = transformer_update_func(simulator, featuer_dict)
    simulator._update_train_series_lead = transformer_lead_update_func(simulator, featuer_dict)

    def _update_train_series(train_series: SliceableTensorDict, self_movements: torch.Tensor, leader_movements: torch.Tensor):
        """
        Update the training series with simulated movements.
        Style is not updated because each style token is fixed before testing.

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
