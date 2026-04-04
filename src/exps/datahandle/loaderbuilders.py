from __future__ import annotations

import inspect
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader

from ...schema import CFNAMES as CF
from ..utils import utils
from ..utils.utils import SampleDataPack
from ..utils.utils_namebuilder import _build_scaler_dict
from . import datasets
from .datapackbuilder import build_dataset
from .data_spliter import load_or_build_split_indices
from .feat_extractor import batch_apply, reaction_time, time_headway
from .filters import CFFilter


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


def _build_core_loaders(d: SampleDataPack, data_config: dict, seed: int) -> tuple[DataLoader, DataLoader, DataLoader, dict]:
    x_groups = data_config["x_groups"]
    y_groups = data_config["y_groups"]

    x_keys = list(x_groups.keys())
    y_keys = list(y_groups.keys())
    x_data = {key: d[:, :, x_groups[key]["features"]] for key in x_keys}
    y_data = {key: d[:, :, y_groups[key]["features"]] for key in y_keys}

    train_idx, val_idx, test_idx = load_or_build_split_indices(d=d, data_config=data_config, seed=seed)
    split_seed = int(data_config.get("split_seed", seed))

    scalers = _build_scaler_dict(x_groups, data_config)
    for key in scalers.keys():
        scalers[key] = datasets._fit_scaler(scalers[key], x_data[key][train_idx])

    transform = datasets.make_transform(scalers, x_groups)
    dataset_cls = data_config["dataset"]
    sample_self_ids = d[:, 0, CF.SELF_ID].astype(np.int64) if CF.SELF_ID in d.names else None

    train_dataset = _build_dataset_split(dataset_cls, x_data, y_data, x_keys, y_keys, train_idx, data_config, transform, sample_self_ids)
    setattr(train_dataset, "source_indices", np.asarray(train_idx, dtype=np.int64))
    val_dataset = _build_dataset_split(dataset_cls, x_data, y_data, x_keys, y_keys, val_idx, data_config, transform, sample_self_ids)
    setattr(val_dataset, "source_indices", np.asarray(val_idx, dtype=np.int64))
    test_dataset = _build_dataset_split(dataset_cls, x_data, y_data, x_keys, y_keys, test_idx, data_config, transform, sample_self_ids)
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


def build_loader(
    d: SampleDataPack,
    d_filters: List[CFFilter],
    d_filter_config: dict,
    data_config: dict,
    seed: int = 42,
    add_style_features: bool = False,
) -> tuple[SampleDataPack, DataLoader, DataLoader, DataLoader, dict]:
    d = build_dataset(d, d_filters, d_filter_config)
    if add_style_features:
        raise ValueError("Use build_style_loader for style features.")
    train_loader, val_loader, test_loader, scalers = _build_core_loaders(d, data_config, seed)
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

    d.append_col(np.tile(time_axis, (num_samples, 1))[:, :, np.newaxis], CF.TIME)
    d.append_col(
        batch_apply(reaction_time, [d[:, :, CF.LEAD_V], d[:, :, CF.SELF_V], d[:, :, CF.TIME]])[:, :, np.newaxis],
        CF.REACT,
    )
    d.append_col(
        batch_apply(time_headway, [d[:, :, CF.DELTA_X] - d[:, :, CF.LEAD_L], d[:, :, CF.SELF_V]]),
        CF.THW,
    )

    if data_config is None:
        raise ValueError("data_config must be provided for style data loader.")

    train_loader, val_loader, test_loader, scalers = _build_core_loaders(d, data_config, seed)
    return d, train_loader, val_loader, test_loader, scalers


__all__ = ["build_loader", "build_style_loader"]
