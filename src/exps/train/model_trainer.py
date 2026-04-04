from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from ...utils.logger import logger
from ..losses import StyleMultiTaskLoss
from ..models.stylecf import StyleTransformer
from .modes import run_style_training_mode


def train_model_with_mode(*args, **kwargs):
    raise NotImplementedError(
        "train_model_with_mode is not implemented in the current codebase. "
        "StyleCF training should use train_stylecf; transformer/lstm generic training needs a dedicated trainer."
    )


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
