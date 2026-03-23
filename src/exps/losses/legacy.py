from __future__ import annotations

from tensordict import TensorDict
import torch

from ...schema import CFNAMES as CF
from .base_losses import spacing_mse_from_acc


class StyleLoss:
    """
    Backward-compatible spacing-only style loss wrapper.
    """

    def __init__(self, y_features: list[str]):
        self.name_dict = {feat: idx for idx, feat in enumerate(y_features)}
        if CF.DELTA_X not in self.name_dict or CF.LEAD_X not in self.name_dict:
            raise KeyError("StyleLoss requires DELTA_X and LEAD_X in y_features")

    def acc_spacing_mse(self, outputs: torch.Tensor | tuple[torch.Tensor, torch.Tensor], y: TensorDict, dt: float) -> torch.Tensor:
        output_accs = outputs[0] if isinstance(outputs, tuple) else outputs
        y_traj = y["y_seq"].rename(None)
        return spacing_mse_from_acc(
            output_accs,
            y_traj,
            dt=float(dt),
            true_deltax_idx=self.name_dict[CF.DELTA_X],
            true_leadx_idx=self.name_dict[CF.LEAD_X],
        )
