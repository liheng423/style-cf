from __future__ import annotations

"""
Backward-compatible loss facade.

New implementations are organized under `src/exps/losses/`.
"""

from .losses import IDMLoss, LossWeights, StyleLoss, StyleMultiTaskLoss

__all__ = [
    "StyleLoss",
    "StyleMultiTaskLoss",
    "LossWeights",
    "IDMLoss",
]
