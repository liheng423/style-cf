from __future__ import annotations

from typing import Mapping

from .exps.configs import (
    data_filter_config,
    filter_names,
    lstm_data_config,
    lstm_model_config,
    lstm_train_config,
    style_data_config,
    style_train_config,
    train_config_meta,
    train_entry_config,
    transformer_train_config,
)
from .exps.datahandle.datasets import LSTMDataset
from .exps.datahandle.loaderbuilders import build_loader, build_style_loader
from .exps.train.model_trainer import train_model_with_mode, train_stylecf
from .schema import FEATNAMES as FEAT
from .utils.logger import logger
from .utils.rawdata_loader import load_datapack


def _load_datapack(train_cfg: Mapping[str, object]):
    datapack, _, _ = load_datapack(str(train_cfg["rawdata_config"]))
    return datapack


def _build_split_training_config(train_cfg: Mapping[str, object]) -> dict[str, object]:
    return {
        "split_mode": str(train_cfg.get("split_mode", "group_self_id")),
        "split_seed": int(train_cfg.get("split_seed", train_cfg.get("seed", 42))),
        "train_ratio": float(train_cfg.get("train_ratio", 0.7)),
        "val_ratio": float(train_cfg.get("val_ratio", 0.15)),
        "test_ratio": float(train_cfg.get("test_ratio", 0.15)),
        "split_index_path": str(train_cfg["split_index_path"]),
    }


def _build_style_training_data_config(train_cfg: Mapping[str, object]) -> dict[str, object]:
    out = dict(style_data_config)
    out.update(_build_split_training_config(train_cfg))

    for key in (
        "style_window_mode",
        "strict_style_window",
        "style_window_before_seconds",
    ):
        if key in train_cfg:
            out[key] = train_cfg[key]

    return out


def _build_lstm_training_data_config(train_cfg: Mapping[str, object]) -> dict[str, object]:
    out = {
        "batch_size": int(lstm_data_config.get("batch_size", 64)),
        "seq_len": int(lstm_data_config["seq_len"]),
        "pred_len": int(lstm_data_config["pred_len"]),
        "stride": int(lstm_data_config.get("stride", 1)),
        "scaler": lstm_data_config["scaler"],
        "dataset": lstm_data_config.get("dataset", LSTMDataset),
        "x_groups": {
            FEAT.INPUTS: {
                "features": list(lstm_data_config["in_features"]),
                "transform": True,
            }
        },
        "y_groups": {
            "y_seq": {
                "features": list(lstm_data_config["eva_features"]),
            }
        },
    }
    out.update(_build_split_training_config(train_cfg))
    return out


def _build_transformer_model_config(train_cfg: Mapping[str, object]) -> dict[str, object]:
    x_groups = style_data_config["x_groups"]
    return {
        "enc_in": len(x_groups["enc_x"]["features"]),
        "dec_in": len(x_groups["dec_x"]["features"]),
        "seq_len": int(style_data_config["seq_len"]),
        "label_len": int(style_data_config["label_len"]),
        "pred_len": int(style_data_config["pred_len"]),
        "num_encoder_layers": int(train_cfg.get("transformer_num_encoder_layers", 1)),
        "num_decoder_layers": int(train_cfg.get("transformer_num_decoder_layers", 1)),
    }


def run_style_train(head: int | None = None):
    d = _load_datapack(style_train_config)
    if head is not None:
        d = d.head(head)

    style_train_data_config = _build_style_training_data_config(style_train_config)
    _, train_loader, val_loader, test_loader, _ = build_style_loader(
        d,
        filter_names,
        data_filter_config,
        data_config=style_train_data_config,
        seed=int(style_train_data_config["split_seed"]),
    )
    return train_stylecf(
        style_data_config,
        style_train_config,
        train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
    )


def run_transformer_train(head: int | None = None):
    from .exps.models.transfollower import Transfollower

    if str(transformer_train_config.get("training_mode", "single_stage")).lower() != "single_stage":
        raise ValueError("transformer only supports training_mode='single_stage'.")

    d = _load_datapack(transformer_train_config)
    if head is not None:
        d = d.head(head)

    style_train_data_config = _build_style_training_data_config(transformer_train_config)
    _, train_loader, val_loader, test_loader, _ = build_style_loader(
        d,
        filter_names,
        data_filter_config,
        data_config=style_train_data_config,
        seed=int(style_train_data_config["split_seed"]),
    )

    model = Transfollower(_build_transformer_model_config(transformer_train_config))
    return train_model_with_mode(
        model=model,
        train_config=transformer_train_config,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        model_name="transformer",
    )


def run_lstm_train(head: int | None = None):
    from .exps.models.lstm import CF_LSTM

    if str(lstm_train_config.get("training_mode", "single_stage")).lower() != "single_stage":
        raise ValueError("lstm only supports training_mode='single_stage'.")

    d = _load_datapack(lstm_train_config)
    if head is not None:
        d = d.head(head)

    lstm_train_data_config = _build_lstm_training_data_config(lstm_train_config)
    _, train_loader, val_loader, test_loader, _ = build_loader(
        d,
        filter_names,
        data_filter_config,
        data_config=lstm_train_data_config,
        add_style_features=False,
        seed=int(lstm_train_data_config["split_seed"]),
    )

    model = CF_LSTM(lstm_model_config)
    return train_model_with_mode(
        model=model,
        train_config=lstm_train_config,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        model_name="lstm",
    )


def run_train_from_config(head: int | None = None):
    target_model = str(train_entry_config.get("target_model", "stylecf")).lower()
    if target_model in {"style", "stylecf"}:
        return run_style_train(head=head)
    if target_model == "transformer":
        return run_transformer_train(head=head)
    if target_model == "lstm":
        return run_lstm_train(head=head)
    raise ValueError(
        f"Unsupported train_entry_config.target_model='{target_model}'. "
        "Expected one of: stylecf, transformer, lstm."
    )


def _log_loaded_training_config() -> None:
    target_model = str(train_entry_config.get("target_model", "stylecf")).lower()
    if target_model in {"style", "stylecf"}:
        cfg_name = "style_train_config"
        cfg = style_train_config
    elif target_model == "transformer":
        cfg_name = "transformer_train_config"
        cfg = transformer_train_config
    elif target_model == "lstm":
        cfg_name = "lstm_train_config"
        cfg = lstm_train_config
    else:
        cfg_name = "unknown"
        cfg = {}

    profile_name = str(train_config_meta.get("profile_name", "<default>"))
    profile_path = str(train_config_meta.get("profile_path", ""))
    active_path = str(train_config_meta.get("active_path", ""))

    logger.info(
        "Loaded training profile | "
        f"active_path={active_path} profile={profile_name} profile_path={profile_path}"
    )
    logger.info(
        "Resolved train config | "
        f"target_model={target_model} cfg={cfg_name} "
        f"training_mode={cfg.get('training_mode')} num_epoch={cfg.get('num_epoch')} "
        f"stage1_epochs={cfg.get('stage1_epochs')} stage2_epochs={cfg.get('stage2_epochs')} stage3_epochs={cfg.get('stage3_epochs')} "
        f"lr={cfg.get('lr')} stage1_lr={cfg.get('stage1_lr')} stage2_lr={cfg.get('stage2_lr')} stage3_lr={cfg.get('stage3_lr')} "
        f"w_spacing={cfg.get('w_spacing')} w_acc={cfg.get('w_acc')} w_contrastive={cfg.get('w_contrastive')} "
        f"split_index_path={cfg.get('split_index_path')}"
    )


def main() -> int:
    _log_loaded_training_config()
    run_train_from_config()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
