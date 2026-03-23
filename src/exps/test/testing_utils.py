from __future__ import annotations

from typing import Mapping, cast

import torch
import torch.nn.functional as F
from tensordict import TensorDict

from ..datahandle import dataset as dataset_utils
from ..datahandle.stylebuilder import build_style_tokens_from_datapack
from ..models.lstm import CF_LSTM
from ..models.stylecf import EmbeddingStyleTransformer, StyleTransformer
from ..models.transfollower import Transfollower
from ..utils.utils import SliceableTensorDict, drop_tensor_names, stack_name
from ...schema import CFNAMES as CF, FEATNAMES as FEAT
from ...stylecf.schema import TensorNames


def concat_time(tensor_list):
    if not tensor_list:
        raise ValueError("tensor_list must be non-empty")
    first = tensor_list[0]
    if isinstance(first, SliceableTensorDict):
        return stack_name(tensor_list, TensorNames.T)
    return torch.concat(tensor_list, dim=0)


def build_batch_input(x: SliceableTensorDict) -> TensorDict:
    payload: dict[str, torch.Tensor] = {}
    for key in x.keys():
        value = drop_tensor_names(cast(torch.Tensor, x[key]))
        if value.ndim in (1, 2):
            value = value.unsqueeze(0)
        payload[key] = value
    return TensorDict(payload, batch_size=[1])


def style_embed_pred_func(model: EmbeddingStyleTransformer, data: SliceableTensorDict, *_):
    pred_acc = model(build_batch_input(data))
    return drop_tensor_names(pred_acc.squeeze(0))


def transformer_pred_func(model: Transfollower, data: SliceableTensorDict, *_):
    pred_acc = model(build_batch_input(data))
    return drop_tensor_names(pred_acc.squeeze(0))


def lstm_pred_func(model: CF_LSTM, data: SliceableTensorDict, *_):
    inputs = drop_tensor_names(cast(torch.Tensor, data[FEAT.INPUTS])).unsqueeze(0)
    pred_acc = model(inputs)
    return drop_tensor_names(pred_acc.squeeze(0))


def build_test_traj_builder(d_test, device: torch.device):
    def _build(idx: int):
        self_traj = torch.tensor(
            d_test[idx, :, [CF.SELF_X, CF.SELF_V, CF.SELF_A]],
            dtype=torch.float32,
            device=device,
        )
        leader_traj = torch.tensor(
            d_test[idx, :, [CF.LEAD_X, CF.LEAD_V, CF.LEAD_A]],
            dtype=torch.float32,
            device=device,
        )
        return self_traj, leader_traj

    return _build


def build_group_input(
    d_test,
    idx: int,
    x_groups: Mapping[str, Mapping[str, object]],
    scalers: Mapping[str, object],
    keys: tuple[str, ...],
    device: torch.device,
) -> SliceableTensorDict:
    payload: dict[str, torch.Tensor] = {}
    for key in keys:
        group = x_groups[key]
        features = cast(list[str], group["features"])
        series = d_test[idx, :, features]

        if bool(group.get("transform", True)) and key in scalers:
            series = dataset_utils._transform(scalers[key], series)

        payload[key] = torch.tensor(series, dtype=torch.float32, device=device).rename(
            TensorNames.T,
            TensorNames.F,
        )

    return SliceableTensorDict(payload, batch_size=[])


def build_style_tokens(
    d_style,
    d_test,
    style_model: StyleTransformer,
    style_feature_names: list[str],
    style_token_seconds: float,
    style_window_before_seconds: tuple[float, float] | None,
    style_token_mode: str,
    style_token_source: str,
    style_token_anchor_step: int | None,
    style_scaler: object | None,
    device: torch.device,
):
    if style_token_source == "style_window_head":
        style_tokens = build_style_tokens_from_datapack(
            d_style,
            style_feature_names,
            style_model.embedder,
            seconds=style_token_seconds,
            window_before_seconds=None,
            scaler=style_scaler,
            device=device,
        )
    elif style_token_source == "nearby_before_start":
        style_tokens = build_style_tokens_from_datapack(
            d_test,
            style_feature_names,
            style_model.embedder,
            seconds=style_token_seconds,
            window_before_seconds=style_window_before_seconds,
            scaler=style_scaler,
            device=device,
            end_step=style_token_anchor_step,
        )
    else:
        raise ValueError(
            "Unsupported style_token_source: "
            f"{style_token_source}. Expected 'style_window_head' or 'nearby_before_start'."
        )

    if style_token_mode == "global_center":
        center = style_tokens.mean(dim=0, keepdim=True)
        style_tokens = center.repeat(d_test.data.shape[0], 1)
    elif style_tokens.shape[0] != d_test.data.shape[0]:
        center = style_tokens.mean(dim=0, keepdim=True)
        style_tokens = center.repeat(d_test.data.shape[0], 1)

    return style_tokens


def build_dummy_style_tokens(
    d_test,
    embed_dim: int,
    style_token_mode: str,
    device: torch.device,
    seed: int | None = 42,
):
    def _normalize_rows(tokens: torch.Tensor) -> torch.Tensor:
        return F.normalize(tokens, p=2, dim=1, eps=1e-12)

    num_samples = int(d_test.data.shape[0])

    if seed is None:
        style_tokens = torch.randn(num_samples, embed_dim, device=device)
    else:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        style_tokens = torch.randn((num_samples, embed_dim), generator=generator).to(device)

    if style_token_mode == "global_center":
        center = style_tokens.mean(dim=0, keepdim=True)
        style_tokens = _normalize_rows(center).repeat(num_samples, 1)
    else:
        style_tokens = _normalize_rows(style_tokens)

    return style_tokens


def shuffle_style_tokens(style_tokens: torch.Tensor, seed: int | None = 42) -> torch.Tensor:
    num_samples = int(style_tokens.shape[0])
    if num_samples <= 1:
        return style_tokens

    if seed is None:
        perm = torch.randperm(num_samples, device=style_tokens.device)
    else:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        perm = torch.randperm(num_samples, generator=generator).to(style_tokens.device)

    identity = torch.arange(num_samples, device=style_tokens.device)
    if torch.equal(perm, identity):
        perm = torch.roll(perm, shifts=1)

    return style_tokens[perm]


def build_transfollower_config(
    x_groups: Mapping[str, Mapping[str, object]],
    seq_len: int,
    label_len: int,
    pred_len: int,
) -> dict[str, int]:
    return {
        "enc_in": len(cast(list[str], x_groups["enc_x"]["features"])),
        "dec_in": len(cast(list[str], x_groups["dec_x"]["features"])),
        "seq_len": seq_len,
        "label_len": label_len,
        "pred_len": pred_len,
    }
