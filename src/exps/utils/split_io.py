from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np


def save_split_indices(
    path: str | Path,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "train_idx": np.asarray(train_idx, dtype=np.int64),
        "val_idx": np.asarray(val_idx, dtype=np.int64),
        "test_idx": np.asarray(test_idx, dtype=np.int64),
    }
    if metadata:
        for key, value in metadata.items():
            payload[f"meta_{key}"] = value

    np.savez(p, **payload)
    return p


def load_split_indices(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Split index file not found: {p}")

    with np.load(p, allow_pickle=True) as data:
        out: dict[str, Any] = {
            "train_idx": data["train_idx"].astype(np.int64),
            "val_idx": data["val_idx"].astype(np.int64),
            "test_idx": data["test_idx"].astype(np.int64),
        }
        for key in data.files:
            if key.startswith("meta_"):
                out[key] = data[key]
    return out


__all__ = ["save_split_indices", "load_split_indices"]
