from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from tensordict import TensorDict

from ...schema import CFNAMES as CF
from .base_losses import acc_mse_from_acc, spacing_mse_from_acc
from .contrastive import supervised_contrastive_loss


@dataclass(frozen=True)
class LossWeights:
    spacing: float = 1.0
    acc: float = 0.3
    contrastive: float = 0.1


class StyleMultiTaskLoss:
    """
    Multi-task training objective:
    L = w_spacing * L_spacing + w_acc * L_acc + w_contrastive * L_contrastive(style)
    """

    def __init__(
        self,
        y_features: list[str],
        default_weights: LossWeights | None = None,
        contrastive_temperature: float = 0.2,
    ) -> None:
        self.name_dict = {feat: idx for idx, feat in enumerate(y_features)}
        for req in (CF.DELTA_X, CF.LEAD_X, CF.SELF_A):
            if req not in self.name_dict:
                raise KeyError(f"Missing required y feature for style loss: {req}")

        self.default_weights = default_weights or LossWeights()
        self.contrastive_temperature = float(contrastive_temperature)

    def _resolve_labels(self, y: TensorDict, labels: torch.Tensor | None = None) -> torch.Tensor | None:
        if labels is not None:
            return labels.view(-1).long()
        if "self_id" not in y.keys():
            return None
        raw = y["self_id"]
        if raw.ndim > 1:
            raw = raw.squeeze(-1)
        return raw.rename(None).view(-1).long()

    def spacing_loss(self, outputs: torch.Tensor | tuple[torch.Tensor, torch.Tensor], y: TensorDict, dt: float) -> torch.Tensor:
        output_accs = outputs[0] if isinstance(outputs, tuple) else outputs
        y_traj = y["y_seq"].rename(None)
        return spacing_mse_from_acc(
            output_accs,
            y_traj,
            dt=float(dt),
            true_deltax_idx=self.name_dict[CF.DELTA_X],
            true_leadx_idx=self.name_dict[CF.LEAD_X],
        )

    def acc_loss(self, outputs: torch.Tensor | tuple[torch.Tensor, torch.Tensor], y: TensorDict) -> torch.Tensor:
        output_accs = outputs[0] if isinstance(outputs, tuple) else outputs
        y_traj = y["y_seq"].rename(None)
        return acc_mse_from_acc(output_accs, y_traj, true_acc_idx=self.name_dict[CF.SELF_A])

    def contrastive_loss(
        self,
        outputs: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        y: TensorDict,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not isinstance(outputs, tuple) or len(outputs) < 2:
            accs = outputs if isinstance(outputs, torch.Tensor) else None
            if accs is None:
                raise ValueError("outputs cannot be None")
            return torch.zeros((), device=accs.device, dtype=accs.dtype)
        _, embeds = outputs
        resolved_labels = self._resolve_labels(y, labels=labels)
        return supervised_contrastive_loss(
            embeddings=embeds,
            labels=resolved_labels,
            temperature=self.contrastive_temperature,
        )

    def compute(
        self,
        outputs: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        y: TensorDict,
        dt: float,
        labels: torch.Tensor | None = None,
        weights: LossWeights | None = None,
    ) -> Mapping[str, torch.Tensor]:
        w = weights or self.default_weights

        spacing = self.spacing_loss(outputs, y, dt)
        acc = self.acc_loss(outputs, y)
        contrastive = self.contrastive_loss(outputs, y, labels=labels)

        total = (
            float(w.spacing) * spacing
            + float(w.acc) * acc
            + float(w.contrastive) * contrastive
        )
        return {
            "total": total,
            "spacing": spacing,
            "acc": acc,
            "contrastive": contrastive,
        }

    def __call__(self, outputs: torch.Tensor | tuple[torch.Tensor, torch.Tensor], y: TensorDict, dt: float) -> torch.Tensor:
        return self.compute(outputs, y, dt)["total"]
