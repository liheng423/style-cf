from __future__ import annotations

import hashlib
import json

"""
Backward-compatible utility facade.

Implementation has been split into focused modules:
- named_tensors.py
- tensordict_ops.py
- datapack.py
- io.py
"""

from .datapack import SampleDataPack, build_id_datapack, load_zen_data
from .io import ensure_dir, model_save
from .named_tensors import (
    drop_tensor_names,
    restore_td_names_like,
    restore_tensor_names_like,
    strip_td_names,
    strip_tensor_names,
)
from .sliceable_tensordict import SliceableTensorDict
from .tensordict_ops import _collate, stack_name, td_cat


def _normalize_for_fingerprint(value):
    if isinstance(value, dict):
        return {str(key): _normalize_for_fingerprint(val) for key, val in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_normalize_for_fingerprint(item) for item in value]
    if isinstance(value, set):
        return [_normalize_for_fingerprint(item) for item in sorted(value, key=repr)]
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes, bytearray)):
        try:
            return _normalize_for_fingerprint(value.tolist())
        except TypeError:
            pass
    if hasattr(value, "item") and not isinstance(value, (str, bytes, bytearray)):
        try:
            return _normalize_for_fingerprint(value.item())
        except (ValueError, TypeError):
            pass
    if hasattr(value, "as_posix"):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "__module__") and hasattr(value, "__qualname__"):
        return f"{value.__module__}.{value.__qualname__}"
    return repr(value)


def fingerprint(*values) -> str:
    payload = _normalize_for_fingerprint(values)
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()

__all__ = [
    "SliceableTensorDict",
    "SampleDataPack",
    "load_zen_data",
    "build_id_datapack",
    "model_save",
    "ensure_dir",
    "drop_tensor_names",
    "strip_tensor_names",
    "restore_tensor_names_like",
    "strip_td_names",
    "restore_td_names_like",
    "_collate",
    "td_cat",
    "stack_name",
    "fingerprint",
]
