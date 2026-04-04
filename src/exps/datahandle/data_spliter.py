from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from ...schema import CFNAMES as CF
from ..utils.datapack import SampleDataPack
from ..utils.split_io import load_split_indices, save_split_indices as persist_split_indices


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


def _fingerprint(*values) -> str:
    payload = _normalize_for_fingerprint(values)
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _resolve_split_ratios(data_config: dict) -> tuple[float, float, float]:
    train_ratio = float(data_config.get("train_ratio", data_config.get("train_data_ratio", 0.7)))
    val_ratio = float(data_config.get("val_ratio", 0.0))
    test_ratio = float(data_config.get("test_ratio", 1.0 - train_ratio - val_ratio))

    if train_ratio <= 0 or test_ratio <= 0:
        raise ValueError(
            f"Invalid split ratios: train={train_ratio}, val={val_ratio}, test={test_ratio}. "
            "train and test must be positive."
        )
    if val_ratio < 0:
        raise ValueError(f"Invalid val_ratio={val_ratio}, val_ratio must be >= 0.")

    total = train_ratio + val_ratio + test_ratio
    if not np.isclose(total, 1.0, atol=1e-6):
        raise ValueError(
            f"Split ratios must sum to 1.0, got train={train_ratio}, val={val_ratio}, test={test_ratio}."
        )
    return train_ratio, val_ratio, test_ratio


def _split_indices_random(
    num_samples: int,
    train_ratio: float,
    val_ratio: float,
    split_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from sklearn.model_selection import train_test_split

    indices = np.arange(num_samples)
    if val_ratio <= 0.0:
        train_idx, test_idx = train_test_split(
            indices,
            test_size=1 - train_ratio,
            random_state=split_seed,
            shuffle=True,
        )
        val_idx = test_idx.copy()
    else:
        holdout_ratio = 1.0 - train_ratio
        train_idx, holdout_idx = train_test_split(
            indices,
            test_size=holdout_ratio,
            random_state=split_seed,
            shuffle=True,
        )
        val_share = val_ratio / holdout_ratio
        val_idx, test_idx = train_test_split(
            holdout_idx,
            test_size=1 - val_share,
            random_state=split_seed,
            shuffle=True,
        )

    if len(train_idx) == 0 or len(val_idx) == 0 or len(test_idx) == 0:
        raise ValueError(
            "Split produced an empty set in random mode. "
            f"train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}."
        )

    return np.sort(train_idx), np.sort(val_idx), np.sort(test_idx)


def _split_indices_group_self_id(
    d: SampleDataPack,
    train_ratio: float,
    val_ratio: float,
    split_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if CF.SELF_ID not in d.names:
        raise KeyError(f"{CF.SELF_ID} is required for split_mode='group_self_id'.")

    self_ids = d[:, 0, CF.SELF_ID].astype(np.int64)
    unique_ids = np.unique(self_ids)
    if unique_ids.shape[0] < 3:
        raise ValueError(
            f"Need at least 3 unique SELF_ID values for train/val/test split, got {unique_ids.shape[0]}."
        )

    rng = np.random.default_rng(split_seed)
    shuffled_ids = unique_ids.copy()
    rng.shuffle(shuffled_ids)

    n_ids = shuffled_ids.shape[0]
    n_train = max(1, int(np.floor(n_ids * train_ratio)))
    n_val = max(1, int(np.floor(n_ids * val_ratio)))
    n_test = n_ids - n_train - n_val

    if n_test <= 0:
        n_test = 1
        if n_val > 1:
            n_val -= 1
        elif n_train > 1:
            n_train -= 1
        else:
            raise ValueError("Unable to allocate non-empty train/val/test SELF_ID groups.")

    train_ids = shuffled_ids[:n_train]
    val_ids = shuffled_ids[n_train : n_train + n_val]
    test_ids = shuffled_ids[n_train + n_val :]

    train_idx = np.flatnonzero(np.isin(self_ids, train_ids))
    val_idx = np.flatnonzero(np.isin(self_ids, val_ids))
    test_idx = np.flatnonzero(np.isin(self_ids, test_ids))

    if len(train_idx) == 0 or len(val_idx) == 0 or len(test_idx) == 0:
        raise ValueError(
            "Group split produced an empty set. "
            f"train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}."
        )

    return np.sort(train_idx), np.sort(val_idx), np.sort(test_idx)


def save_split_indices(
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    save_path: str | Path | None,
    split_config: dict | None = None,
):
    if save_path in (None, ""):
        return None

    path = Path(save_path)
    normalized_path = str(path.expanduser().resolve(strict=False))
    metadata = {
        "fingerprint": _fingerprint(normalized_path, split_config),
        "fingerprint_path": normalized_path,
    }
    if split_config is not None:
        metadata["split_config"] = np.array(split_config, dtype=object)

    return persist_split_indices(
        path=path,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        metadata=metadata,
    )


def build_split_indices(
    d: SampleDataPack,
    data_config: dict,
    seed: int,
    save: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_ratio, val_ratio, _ = _resolve_split_ratios(data_config)
    split_seed = int(data_config.get("split_seed", seed))
    split_mode = str(data_config.get("split_mode", "random")).lower()

    if split_mode == "group_self_id":
        train_idx, val_idx, test_idx = _split_indices_group_self_id(
            d=d,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            split_seed=split_seed,
        )
    else:
        train_idx, val_idx, test_idx = _split_indices_random(
            num_samples=int(d.data.shape[0]),
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            split_seed=split_seed,
        )

    if save:
        save_split_indices(
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            save_path=data_config.get("split_index_path"),
            split_config=data_config,
        )

    return train_idx, val_idx, test_idx


def load_saved_split_indices(
    save_path: str | Path,
    split_config: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    payload = load_split_indices(save_path)
    train_idx = np.asarray(payload["train_idx"], dtype=np.int64)
    val_idx = np.asarray(payload["val_idx"], dtype=np.int64)
    test_idx = np.asarray(payload["test_idx"], dtype=np.int64)

    if split_config is not None:
        normalized_path = str(Path(save_path).expanduser().resolve(strict=False))
        expected = _fingerprint(normalized_path, split_config)
        actual = payload.get("meta_fingerprint")
        if isinstance(actual, np.ndarray):
            actual = actual.item()
        if actual is not None and str(actual) != expected:
            raise ValueError(
                f"Split fingerprint mismatch for {save_path}. "
                "The saved indices were not generated from the current path/config."
            )

    return train_idx, val_idx, test_idx


def load_or_build_split_indices(
    d: SampleDataPack,
    data_config: dict,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    save_path = data_config.get("split_index_path")
    if save_path not in (None, ""):
        path = Path(save_path)
        if path.exists():
            return load_saved_split_indices(path, split_config=data_config)

    return build_split_indices(d=d, data_config=data_config, seed=seed, save=True)
