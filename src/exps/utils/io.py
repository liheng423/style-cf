from __future__ import annotations

import os

import torch


def model_save(model_dict, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = model_dict.state_dict() if hasattr(model_dict, "state_dict") else model_dict
    torch.save(payload, path)
    return path


def ensure_dir(folder):
    """Ensure the folder exists."""
    os.makedirs(folder, exist_ok=True)


__all__ = ["ensure_dir", "model_save"]
