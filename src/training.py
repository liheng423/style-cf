from __future__ import annotations

import os

from .exps.configs import data_filter_config, filter_names, style_data_config, style_train_config
from .exps.train.model_trainer import build_style_loader, train_stylecf
from .exps.utils.scaler_io import save_scaler_payload
from .exps.utils.split_io import save_split_indices
from .exps.utils.utils import load_zen_data
from .utils.logger import logger


def _resolve_data_path() -> str:
    env_path = os.environ.get("ZEN_DATA_PATH")
    cfg_path = style_train_config.get("datapath")
    path = env_path or cfg_path
    if not path:
        raise ValueError("Training data path is missing. Set ZEN_DATA_PATH or style_train_config['datapath'].")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Training dataset not found: {path}")
    return str(path)


def _dataset(head: int | None = None):
    d = load_zen_data(_resolve_data_path(), rise=True, in_kph=False, kilo_norm=True)
    return d.head(head) if head is not None else d


def _save_training_scalers(scalers: dict[str, object]) -> None:
    style_path = str(style_train_config.get("style_scaler_path", "data/saved_scalers/scaler_stylecf.pkl"))
    transformer_path = str(
        style_train_config.get("transformer_scaler_path", "data/saved_scalers/scaler_transfollower.pkl")
    )

    style_payload = dict(scalers)
    transformer_payload = {
        key: style_payload[key]
        for key in ("enc_x", "dec_x")
        if key in style_payload
    }

    style_saved = save_scaler_payload(style_payload, style_path)
    logger.info(f"Saved style scalers to {style_saved}")

    if transformer_payload:
        transformer_saved = save_scaler_payload(transformer_payload, transformer_path)
        logger.info(f"Saved transformer scalers to {transformer_saved}")


def _save_training_split(train_loader, val_loader, test_loader) -> None:
    split_path = str(style_train_config.get("split_index_path", "data/saved_splits/stylecf_group_split.npz"))

    train_idx = getattr(train_loader.dataset, "source_indices", None)
    val_idx = getattr(val_loader.dataset, "source_indices", None)
    test_idx = getattr(test_loader.dataset, "source_indices", None)
    if train_idx is None or val_idx is None or test_idx is None:
        logger.warning("Skip saving split indices: source_indices missing on dataset.")
        return

    saved = save_split_indices(
        split_path,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        metadata={
            "split_mode": style_data_config.get("split_mode", "random"),
            "split_seed": style_data_config.get("split_seed", style_train_config.get("seed", 42)),
            "train_ratio": style_data_config.get("train_ratio", style_data_config.get("train_data_ratio", 0.7)),
            "val_ratio": style_data_config.get("val_ratio", 0.0),
            "test_ratio": style_data_config.get("test_ratio", 0.0),
        },
    )
    logger.info(f"Saved train/val/test split indices to {saved}")


def run_style_train(head: int | None = None):
    d = _dataset(head=head)
    _, train_loader, val_loader, test_loader, scalers = build_style_loader(
        d,
        filter_names,
        data_filter_config,
        data_config=style_data_config,
        seed=int(style_data_config.get("split_seed", style_train_config.get("seed", 42))),
    )
    _save_training_scalers(scalers)
    _save_training_split(train_loader, val_loader, test_loader)
    return train_stylecf(
        style_data_config,
        style_train_config,
        train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
    )


def main() -> int:
    run_style_train()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
