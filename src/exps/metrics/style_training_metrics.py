from __future__ import annotations

from typing import Mapping

import torch
from tensordict import TensorDict

from ..losses.style_losses import StyleMultiTaskLoss


def compute_shuffle_gap(
    model,
    x: TensorDict,
    y: TensorDict,
    criterion: StyleMultiTaskLoss,
    dt: float,
    eps: float = 1e-12,
) -> float:
    """
    Shuffle Gap = (MSE_shuffle - MSE_style) / MSE_style
    """
    style = x["style"]
    batch_size = int(style.shape[0])
    if batch_size <= 1:
        return float("nan")

    device = style.device
    perm = torch.randperm(batch_size, device=device)
    if torch.equal(perm, torch.arange(batch_size, device=device)):
        perm = torch.roll(perm, shifts=1)

    out_style = model(x)
    mse_style = criterion.spacing_loss(out_style, y, dt)

    shuffled_style = style.rename(None)[perm]
    if style.names is not None:
        shuffled_style = shuffled_style.refine_names(*style.names)

    x_shuffle = x.clone()
    x_shuffle["style"] = shuffled_style
    out_shuffle = model(x_shuffle)
    mse_shuffle = criterion.spacing_loss(out_shuffle, y, dt)

    denom = float(mse_style.detach().item()) + eps
    return (float(mse_shuffle.detach().item()) - float(mse_style.detach().item())) / denom


def compute_embedding_distance_stats(
    embeddings: torch.Tensor | None,
    labels: torch.Tensor | None,
) -> Mapping[str, float]:
    if embeddings is None or labels is None:
        return {
            "embedding_intra": float("nan"),
            "embedding_inter": float("nan"),
            "embedding_ratio": float("nan"),
        }

    labels = labels.view(-1).long()
    if embeddings.ndim != 2 or embeddings.shape[0] != labels.shape[0]:
        raise ValueError(
            "Embedding/label shape mismatch: "
            f"embeddings={tuple(embeddings.shape)}, labels={tuple(labels.shape)}"
        )
    if embeddings.shape[0] <= 1:
        return {
            "embedding_intra": float("nan"),
            "embedding_inter": float("nan"),
            "embedding_ratio": float("nan"),
        }

    dists = torch.cdist(embeddings, embeddings, p=2)
    identity = torch.eye(dists.shape[0], dtype=torch.bool, device=dists.device)
    same = (labels.unsqueeze(0) == labels.unsqueeze(1)) & (~identity)
    diff = (labels.unsqueeze(0) != labels.unsqueeze(1))

    intra = float(dists[same].mean().item()) if bool(same.any()) else float("nan")
    inter = float(dists[diff].mean().item()) if bool(diff.any()) else float("nan")
    ratio = inter / max(intra, 1e-12) if intra == intra and inter == inter else float("nan")
    return {
        "embedding_intra": intra,
        "embedding_inter": inter,
        "embedding_ratio": ratio,
    }
