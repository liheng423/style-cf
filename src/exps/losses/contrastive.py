from __future__ import annotations

import torch
import torch.nn.functional as F


def supervised_contrastive_loss(
    embeddings: torch.Tensor | None,
    labels: torch.Tensor | None,
    temperature: float = 0.2,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Supervised contrastive loss with in-batch positives defined by equal labels.

    Returns zero when no valid positive pairs exist in the batch.
    """
    if embeddings is None or labels is None:
        if embeddings is None:
            raise ValueError("embeddings cannot be None when computing contrastive loss")
        return torch.zeros((), device=embeddings.device, dtype=embeddings.dtype)

    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must have shape (B, D), got {tuple(embeddings.shape)}")

    labels = labels.view(-1).long()
    if labels.shape[0] != embeddings.shape[0]:
        raise ValueError(
            "labels and embeddings batch size mismatch: "
            f"{labels.shape[0]} vs {embeddings.shape[0]}"
        )

    if embeddings.shape[0] <= 1:
        return torch.zeros((), device=embeddings.device, dtype=embeddings.dtype)

    z = F.normalize(embeddings, p=2, dim=1)
    logits = torch.matmul(z, z.T) / max(float(temperature), eps)
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()

    identity = torch.eye(logits.shape[0], dtype=torch.bool, device=logits.device)
    pair_mask = ~identity
    positive_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)) & pair_mask

    positives_per_row = positive_mask.sum(dim=1)
    valid_row_mask = positives_per_row > 0
    if not bool(valid_row_mask.any()):
        return torch.zeros((), device=embeddings.device, dtype=embeddings.dtype)

    exp_logits = torch.exp(logits) * pair_mask
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + eps)

    mean_log_prob_pos = (positive_mask * log_prob).sum(dim=1) / positives_per_row.clamp_min(1)
    loss = -mean_log_prob_pos[valid_row_mask].mean()
    return loss
