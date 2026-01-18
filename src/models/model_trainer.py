
from typing import List
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tensordict import TensorDict
from src.models import utils

from src.models.filters import CFFilter
from src.models.style_cf import batch_apply, reaction_time, time_headway
from src.models.utils import SampleDataPack
from src.models import dataset
from src.schema import CFNAMES as CF


def build_dataset(d: SampleDataPack, d_filters: List[CFFilter], d_filter_config: dict) -> SampleDataPack:
    """
    Build the dataset for car-following model training.
    """
    d = d.normalize_kilopost()  # normalize KILO to rising
    d.append_col(d[:, :, CF.LEAD_V] - d[:, :, CF.SELF_V], CF.DELTA_V)
    d.append_col(d[:, :, CF.LEAD_X] - d[:, :, CF.SELF_X], CF.DELTA_X)
    d_filter = CFFilter(d, d_filter_config)
    d = d_filter.filter(d_filters)

    d.force_consistent()

    

    return d

def build_style_dataset(
    d: SampleDataPack,
    d_filters: List[CFFilter],
    d_filter_config: dict,
    data_config: dict | None = None,
    seed: int = 42,
) -> tuple[SampleDataPack, DataLoader, DataLoader, list]:
    """
    Build the dataset for style-based car-following model training.
    """
    d = build_dataset(d, d_filters, d_filter_config)
    
    # construct new features
    d.append_col(
        np.expand_dims(
            np.tile(np.arange(0, 30, 0.1), (len(d[:, 0, CF.LEAD_V]), 1)),
            2,
        ),
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
            [d[:, :, CF.DELTA_X] - d[:, :, CF.LEAD_L], d[:, :, CF.SELF_V]], # use net distance
        ),
        CF.THW,
    )

    if data_config is None:
        return d
    train_loader, test_loader, scalers = pipeline(d, data_config, seed)
    return d, train_loader, test_loader, scalers


def pipeline(d: SampleDataPack, data_config: dict, seed: int) -> tuple[DataLoader, DataLoader, list]:
    x_groups = data_config["x_groups"]
    y_groups = data_config["y_groups"]

    x_data = [d[:, :, group["features"]] for group in x_groups]
    y_data = [d[:, :, group["features"]] for group in y_groups]

    num_samples = y_data[0].shape[0]
    indices = np.arange(num_samples)

    train_idx, test_idx = train_test_split(
        indices,
        test_size=1 - data_config["train_data_ratio"],
        random_state=seed,
    )

    x_train = [xi[train_idx] for xi in x_data]
    x_test = [xi[test_idx] for xi in x_data]
    y_train = [yi[train_idx] for yi in y_data]
    y_test = [yi[test_idx] for yi in y_data]

    scalers = [data_config["scaler"]() for _ in x_train]

    for data_idx in range(len(scalers)):
        scalers[data_idx] = dataset._fit_scaler(scalers[data_idx], x_train[data_idx])
    transform = dataset.make_transform(scalers, x_groups)

    dataset_cls = data_config["dataset"]
    train_dataset = dataset_cls(*x_train, *y_train, data_config=data_config, transform=transform)
    test_dataset = dataset_cls(*x_test, *y_test, data_config=data_config, transform=transform)
    batch_size = data_config.get("batch_size", 64)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=utils._collate,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        collate_fn=utils._collate,
    )
    return train_loader, test_loader, scalers
