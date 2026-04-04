from __future__ import annotations

import torch
import torch.nn as nn
from typing import Iterable

from . import datasets as dataset_utils
from ...utils.logger import logger
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
    window_before_seconds: tuple[float, float] | None = None,
    scaler: object | None = None,
    device: torch.device | None = None,
    end_step: int | None = None,
    batch_size: int | None = None,
    log_every_batches: int | None = None,
) -> torch.Tensor:
    """
    Build style tokens from the first `seconds` of each sample in a SampleDataPack.

    Args:
        datapack: SampleDataPack with data shape (N, time, feature)
        feature_names: feature keys used as style inputs
        embedder: stylecf embedder module
        seconds: number of seconds from the start of each sample to use
        scaler: optional scaler matching training-time style transform
        device: optional device for embedding
        window_before_seconds: optional (near, far) window in seconds before
            `end_step` to use, i.e. [end_step-far, end_step-near). When set,
            it overrides `seconds`.
        end_step: optional exclusive end index for style window anchor.
            If None and window_before_seconds is not set, use leading window.

    Returns:
        style_tokens: torch.Tensor of shape (N, embed_dim)
    """
    if seconds <= 0:
        raise ValueError("seconds must be positive")

    total_steps = int(datapack.data.shape[1])
    if window_before_seconds is not None:
        near_s, far_s = window_before_seconds
        near_s, far_s = min(float(near_s), float(far_s)), max(float(near_s), float(far_s))
        near_steps = max(1, int(round(near_s / datapack.dt)))
        far_steps = max(near_steps + 1, int(round(far_s / datapack.dt)))

        anchor = total_steps if end_step is None else max(1, min(int(end_step), total_steps))
        start = anchor - far_steps
        end = anchor - near_steps

        if start < 0:
            shift = -start
            start = 0
            end = min(total_steps, end + shift)
        if end <= start:
            end = min(total_steps, start + 1)
    else:
        steps = max(1, int(round(seconds / datapack.dt)))
        steps = min(steps, total_steps)

        if end_step is None:
            start = 0
            end = steps
        else:
            end = max(1, min(int(end_step), total_steps))
            start = max(0, end - steps)
            if end <= start:
                end = min(total_steps, start + 1)

    feat_indices = [datapack.names[name] for name in feature_names]
    num_samples = int(datapack.data.shape[0])
    effective_batch_size = int(batch_size or num_samples)
    if effective_batch_size <= 0:
        raise ValueError("batch_size must be positive")

    effective_log_every = int(log_every_batches or 0)
    num_batches = (num_samples + effective_batch_size - 1) // effective_batch_size
    outputs: list[torch.Tensor] = []

    was_training = bool(embedder.training)
    embedder.eval()
    logger.info(
        "Build style embeddings | "
        f"samples={num_samples} window=({start}, {end}) features={len(feat_indices)} "
        f"batch_size={effective_batch_size} batches={num_batches}"
    )

    try:
        with torch.inference_mode():
            for batch_idx, batch_start in enumerate(range(0, num_samples, effective_batch_size), start=1):
                batch_end = min(batch_start + effective_batch_size, num_samples)
                style_traj = datapack.data[batch_start:batch_end, start:end, :][:, :, feat_indices]
                if scaler is not None:
                    style_traj = dataset_utils._transform(scaler, style_traj)

                style_traj_t = torch.tensor(style_traj, dtype=torch.float32)
                if device is not None:
                    style_traj_t = style_traj_t.to(device)

                style_token = build_style(style_traj_t, embedder).detach().cpu()
                outputs.append(style_token)

                if effective_log_every > 0 and (batch_idx == 1 or batch_idx % effective_log_every == 0 or batch_idx == num_batches):
                    logger.info(
                        "Build style embeddings batch | "
                        f"{batch_idx}/{num_batches} samples={batch_end}/{num_samples}"
                    )
    finally:
        if was_training:
            embedder.train()

    return torch.cat(outputs, dim=0)
