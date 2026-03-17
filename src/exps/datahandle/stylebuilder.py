from __future__ import annotations

import torch
import torch.nn as nn
from typing import Iterable

from ..utils.utils import SampleDataPack


def _drop_names(tensor: torch.Tensor) -> torch.Tensor:
    if getattr(tensor, "names", None) is None:
        return tensor
    return tensor.rename(None)


def build_style(data: torch.Tensor, embedder: nn.Module) -> torch.Tensor:
    """
    Build style token from input data by aggregating over time.

    Args:
        data: torch.Tensor of shape (N, time, feature)
    Returns:
        style_token: torch.Tensor of shape (N, 2 * feature)
            (concatenation of per-feature mean and std over time)
    """
    data = _drop_names(data)
    if data.ndim != 3:
        raise ValueError(f"data must be 3D (N, time, feature), got shape {tuple(data.shape)}")

    # Let the stylecf embedder produce the style token.
    style_token = embedder(data)
    return style_token


def build_style_tokens_from_datapack(
    datapack: SampleDataPack,
    feature_names: Iterable[str],
    embedder: nn.Module,
    seconds: float,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Build style tokens from the first `seconds` of each sample in a SampleDataPack.

    Args:
        datapack: SampleDataPack with data shape (N, time, feature)
        feature_names: feature keys used as style inputs
        embedder: stylecf embedder module
        seconds: number of seconds from the start of each sample to use
        device: optional device for embedding

    Returns:
        style_tokens: torch.Tensor of shape (N, embed_dim)
    """
    if seconds <= 0:
        raise ValueError("seconds must be positive")

    steps = max(1, int(round(seconds / datapack.dt)))
    steps = min(steps, datapack.data.shape[1])

    feat_indices = [datapack.names[name] for name in feature_names]
    style_traj = datapack.data[:, :steps, :][:, :, feat_indices]
    style_traj_t = torch.tensor(style_traj, dtype=torch.float32)
    if device is not None:
        style_traj_t = style_traj_t.to(device)

    with torch.no_grad():
        return build_style(style_traj_t, embedder)
