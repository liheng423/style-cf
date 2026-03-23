from __future__ import annotations

import inspect
from typing import List

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from ...schema import CFNAMES as CF
from ...utils.logger import logger
from ..datahandle import dataset
from ..datahandle.databuilder import build_dataset
from ..datahandle.feat_extractor import batch_apply, reaction_time, time_headway
from ..datahandle.filters import CFFilter
from ..losses import StyleMultiTaskLoss
from ..models.stylecf import StyleTransformer
from ..utils import utils
from ..utils.utils import SampleDataPack
from ..utils.utils_namebuilder import _build_scaler_dict
from .modes import run_style_training_mode


def build_loader(
    d: SampleDataPack,
    d_filters: List[CFFilter],
    d_filter_config: dict,
    data_config: dict,
    seed: int = 42,
) -> tuple[SampleDataPack, DataLoader, DataLoader, DataLoader, dict]:
    d = build_dataset(d, d_filters, d_filter_config)
    train_loader, val_loader, test_loader, scalers = pipeline(d, data_config, seed)
    return d, train_loader, val_loader, test_loader, scalers


def build_style_loader(
    d: SampleDataPack,
    d_filters: List[CFFilter],
    d_filter_config: dict,
    data_config: dict | None = None,
    seed: int = 42,
) -> tuple[SampleDataPack, DataLoader, DataLoader, DataLoader, dict]:
    d = build_dataset(d, d_filters, d_filter_config)

    num_samples = d.data.shape[0]
    num_steps = d.data.shape[1]
    time_axis = np.arange(num_steps, dtype=np.float32) * float(d.dt)

    d.append_col(
        np.tile(time_axis, (num_samples, 1))[:, :, np.newaxis],
        CF.TIME,
    )
    d.append_col(
        batch_apply(
            reaction_time,
            [d[:, :, CF.LEAD_V], d[:, :, CF.SELF_V], d[:, :, CF.TIME]],
        )[:, :, np.newaxis],
        CF.REACT,
    )
    d.append_col(
        batch_apply(
            time_headway,
            [d[:, :, CF.DELTA_X] - d[:, :, CF.LEAD_L], d[:, :, CF.SELF_V]],
        ),
        CF.THW,
    )

    if data_config is None:
        raise ValueError("data_config must be provided for style data loader.")

    train_loader, val_loader, test_loader, scalers = pipeline(d, data_config, seed)
    return d, train_loader, val_loader, test_loader, scalers


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
    test_ratio: float,
    split_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        holdout_ratio = val_ratio + test_ratio
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
    test_ratio: float,
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

    train_id_set = set(train_ids.tolist())
    val_id_set = set(val_ids.tolist())
    test_id_set = set(test_ids.tolist())

    if train_id_set & val_id_set or train_id_set & test_id_set or val_id_set & test_id_set:
        raise AssertionError("SELF_ID overlap detected across split groups.")

    train_idx = np.flatnonzero(np.isin(self_ids, train_ids))
    val_idx = np.flatnonzero(np.isin(self_ids, val_ids))
    test_idx = np.flatnonzero(np.isin(self_ids, test_ids))

    if len(train_idx) == 0 or len(val_idx) == 0 or len(test_idx) == 0:
        raise ValueError(
            "Group split produced an empty set. "
            f"train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}."
        )

    return np.sort(train_idx), np.sort(val_idx), np.sort(test_idx)


def _build_dataset_split(
    dataset_cls,
    x_data: dict[str, np.ndarray],
    y_data: dict[str, np.ndarray],
    x_keys: list[str],
    y_keys: list[str],
    idx: np.ndarray,
    data_config: dict,
    transform,
    sample_self_ids: np.ndarray | None,
):
    params = inspect.signature(dataset_cls.__init__).parameters
    kwargs = {}
    if "data_config" in params:
        kwargs["data_config"] = data_config
    if "transform" in params:
        kwargs["transform"] = transform

    if "sample_self_ids" in params and sample_self_ids is not None:
        kwargs["sample_self_ids"] = sample_self_ids[idx]

    return dataset_cls(
        *[x_data[key][idx] for key in x_keys],
        *[y_data[key][idx] for key in y_keys],
        **kwargs,
    )


def pipeline(d: SampleDataPack, data_config: dict, seed: int) -> tuple[DataLoader, DataLoader, DataLoader, dict]:
    x_groups = data_config["x_groups"]
    y_groups = data_config["y_groups"]

    x_keys = list(x_groups.keys())
    y_keys = list(y_groups.keys())
    x_data = {key: d[:, :, x_groups[key]["features"]] for key in x_keys}
    y_data = {key: d[:, :, y_groups[key]["features"]] for key in y_keys}

    train_ratio, val_ratio, test_ratio = _resolve_split_ratios(data_config)
    split_seed = int(data_config.get("split_seed", seed))
    split_mode = str(data_config.get("split_mode", "random")).lower()

    if split_mode == "group_self_id":
        train_idx, val_idx, test_idx = _split_indices_group_self_id(
            d=d,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            split_seed=split_seed,
        )
    else:
        train_idx, val_idx, test_idx = _split_indices_random(
            num_samples=next(iter(y_data.values())).shape[0],
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            split_seed=split_seed,
        )

    scalers = _build_scaler_dict(x_groups, data_config)
    for key in scalers.keys():
        scalers[key] = dataset._fit_scaler(scalers[key], x_data[key][train_idx])

    transform = dataset.make_transform(scalers, x_groups)
    dataset_cls = data_config["dataset"]

    sample_self_ids = d[:, 0, CF.SELF_ID].astype(np.int64) if CF.SELF_ID in d.names else None

    train_dataset = _build_dataset_split(
        dataset_cls,
        x_data,
        y_data,
        x_keys,
        y_keys,
        train_idx,
        data_config,
        transform,
        sample_self_ids,
    )
    setattr(train_dataset, "source_indices", np.asarray(train_idx, dtype=np.int64))
    val_dataset = _build_dataset_split(
        dataset_cls,
        x_data,
        y_data,
        x_keys,
        y_keys,
        val_idx,
        data_config,
        transform,
        sample_self_ids,
    )
    setattr(val_dataset, "source_indices", np.asarray(val_idx, dtype=np.int64))
    test_dataset = _build_dataset_split(
        dataset_cls,
        x_data,
        y_data,
        x_keys,
        y_keys,
        test_idx,
        data_config,
        transform,
        sample_self_ids,
    )
    setattr(test_dataset, "source_indices", np.asarray(test_idx, dtype=np.int64))

    batch_size = data_config.get("batch_size", 64)
    train_gen = torch.Generator()
    train_gen.manual_seed(split_seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=train_gen,
        drop_last=True,
        collate_fn=utils._collate,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=utils._collate,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=utils._collate,
    )

    return train_loader, val_loader, test_loader, scalers


def train_stylecf(
    model_config: dict,
    train_config: dict,
    train_loader: DataLoader,
    val_loader: DataLoader | None = None,
    test_loader: DataLoader | None = None,
):
    model = model_config["model_name"](model_config)

    cfg_device = train_config.get("device", "auto")
    if cfg_device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(cfg_device, torch.device):
        device = cfg_device
    else:
        device = torch.device(str(cfg_device))

    model = model.to(device)

    if val_loader is None:
        val_loader = test_loader
    if val_loader is None:
        raise ValueError("val_loader is required for training.")
    if test_loader is None:
        test_loader = val_loader

    train_dataset = getattr(train_loader, "dataset", None)
    val_dataset = getattr(val_loader, "dataset", None)
    test_dataset = getattr(test_loader, "dataset", None)

    train_traj = int(getattr(train_dataset, "num_samples", -1)) if train_dataset is not None else -1
    val_traj = int(getattr(val_dataset, "num_samples", -1)) if val_dataset is not None else -1
    test_traj = int(getattr(test_dataset, "num_samples", -1)) if test_dataset is not None else -1

    train_windows = len(train_dataset) if train_dataset is not None else -1
    val_windows = len(val_dataset) if val_dataset is not None else -1
    test_windows = len(test_dataset) if test_dataset is not None else -1

    train_batches = len(train_loader)
    val_batches = len(val_loader)
    test_batches = len(test_loader)
    batch_size = int(getattr(train_loader, "batch_size", 0) or 0)
    train_seen_per_epoch = train_batches * batch_size if batch_size > 0 else -1

    logger.info(
        "Data participation before training | "
        f"train_traj={train_traj} val_traj={val_traj} test_traj={test_traj} | "
        f"train_windows={train_windows} val_windows={val_windows} test_windows={test_windows} | "
        f"batch_size={batch_size} train_batches={train_batches} val_batches={val_batches} test_batches={test_batches} | "
        f"train_seen_per_epoch={train_seen_per_epoch}"
    )

    criterion = train_config["loss_func"]
    if not isinstance(criterion, StyleMultiTaskLoss):
        raise TypeError(
            "style_train_config['loss_func'] must resolve to StyleMultiTaskLoss. "
            f"Got: {type(criterion)}"
        )

    logger.info(
        "Starting StyleCF training "
        f"mode={train_config.get('training_mode', 'single_stage')} device={device}"
    )
    assert isinstance(model, StyleTransformer)

    model = run_style_training_mode(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        criterion=criterion,
        train_config=train_config,
        device=device,
    )
    return model
