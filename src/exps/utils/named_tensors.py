from __future__ import annotations

from typing import Any

import torch
from tensordict import TensorDictBase

from .sliceable_tensordict import SliceableTensorDict


def strip_tensor_names(tensor: torch.Tensor) -> torch.Tensor:
    """Return an unnamed tensor for ops that do not support named tensors."""
    if getattr(tensor, "names", None) is None:
        return tensor
    return tensor.rename(None)


def drop_tensor_names(tensor: torch.Tensor) -> torch.Tensor:
    """Backward-compatible alias for `strip_tensor_names`."""
    return strip_tensor_names(tensor)


def restore_tensor_names_like(tensor: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Restore names from `reference` when rank matches."""
    names = getattr(reference, "names", None)
    if names is None or tensor.ndim != len(names):
        return tensor
    return tensor.refine_names(*names)


def strip_td_names(td: TensorDictBase) -> SliceableTensorDict:
    """Strip names from all tensor leaves in a TensorDict-like container."""
    payload: dict[str, Any] = {}
    for key, value in td.items():
        if isinstance(value, torch.Tensor):
            payload[key] = strip_tensor_names(value)
        elif isinstance(value, TensorDictBase):
            payload[key] = strip_td_names(value)
        else:
            payload[key] = value
    return SliceableTensorDict(payload, batch_size=td.batch_size, names=td.names)


def restore_td_names_like(td: TensorDictBase, reference: TensorDictBase) -> SliceableTensorDict:
    """Restore tensor names using a reference TensorDict with the same key structure."""
    payload: dict[str, Any] = {}
    for key, value in td.items():
        ref_value = reference.get(key, None)
        if isinstance(value, torch.Tensor) and isinstance(ref_value, torch.Tensor):
            payload[key] = restore_tensor_names_like(value, ref_value)
        elif isinstance(value, TensorDictBase) and isinstance(ref_value, TensorDictBase):
            payload[key] = restore_td_names_like(value, ref_value)
        else:
            payload[key] = value
    return SliceableTensorDict(payload, batch_size=td.batch_size, names=td.names)


__all__ = [
    "drop_tensor_names",
    "strip_tensor_names",
    "restore_tensor_names_like",
    "strip_td_names",
    "restore_td_names_like",
]
