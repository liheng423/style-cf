from __future__ import annotations

import torch

from .base_losses import distance_mse_from_acc


class IDMLoss:
    def acc_dis_mse(self, outputs: torch.Tensor, y: torch.Tensor, dt: float) -> torch.Tensor:
        outputs = outputs.unsqueeze(0)
        accs_outputs = outputs[..., 2]
        y = y.unsqueeze(0)
        return distance_mse_from_acc(accs_outputs, y, dt)
