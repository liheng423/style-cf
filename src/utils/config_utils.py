from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler

try:
    import tomllib  # type: ignore[attr-defined]
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib  # type: ignore[no-redef]

from ..exps.datahandle.dataset import LSTMDataset, StyledTransfollowerDataset
from ..exps.datahandle.datascalers import DataScaler
from ..exps.losses import IDMLoss, LossWeights, StyleMultiTaskLoss
from ..exps.models.idm import DEFAULT_MASK, DEFAULT_PRED_FUNC, IDM, idm_concat, idm_update_func
from ..exps.models.lstm import CF_LSTM, lstm_update_func
from ..exps.models.stylecf import StyleTransformer, style_update_func, stylecf_mask, transformer_mask
from ..exps.models.transfollower import Transfollower
from ..exps.utils.utils import stack_name
from ..schema import CFNAMES as CF


_CONFIG_DIR = Path(__file__).resolve().parents[1] / "exps" / "config" / "default_configs"
_TESTING_CONFIG_DIR = Path(__file__).resolve().parents[1] / "exps" / "config" / "testing"
_CACHE: dict[str, Any] | None = None


def _load_toml_path(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        return tomllib.load(f)


def _load_toml(name: str) -> dict[str, Any]:
    return _load_toml_path(_CONFIG_DIR / name)


def _deep_merge_dict(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = dict(base)
    for key, value in override.items():
        existing = merged.get(key)
        if isinstance(existing, Mapping) and isinstance(value, Mapping):
            merged[key] = _deep_merge_dict(existing, value)
        else:
            merged[key] = value
    return merged


def _load_test_toml() -> dict[str, Any]:
    """
    Load testing config from experiment selector when available.

    Priority:
    1) src/exps/config/testing/active.toml -> selected experiment file
    2) merge selected experiment overrides onto src/exps/config/default_configs/test.toml
    3) fallback to src/exps/config/default_configs/test.toml
    """
    base_payload = _load_toml("test.toml")

    active_path = _TESTING_CONFIG_DIR / "active.toml"
    if active_path.exists():
        try:
            active_payload = _load_toml_path(active_path)
            testing = active_payload.get("testing", {})
            active_name = testing.get("active") if isinstance(testing, Mapping) else None
            if isinstance(active_name, str) and active_name.strip():
                filename = active_name.strip()
                if not filename.lower().endswith(".toml"):
                    filename = f"{filename}.toml"
                # Keep experiment selection within testing config directory.
                filename = Path(filename).name
                exp_path = _TESTING_CONFIG_DIR / filename
                if exp_path.exists():
                    exp_payload = _load_toml_path(exp_path)
                    if "test_config" in exp_payload and isinstance(exp_payload["test_config"], Mapping):
                        return _deep_merge_dict(base_payload, exp_payload)
        except Exception:
            # Fall back to legacy config path on any active-config parsing issue.
            pass

    return base_payload


def _feature_name(name: str) -> str:
    return getattr(CF, name, name)


def _resolve_features(features: list[str]) -> list[str]:
    return [_feature_name(name) for name in features]


def _resolve_groups(groups: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for group_name, group in groups.items():
        entry = dict(group)
        if "features" in entry:
            entry["features"] = _resolve_features(list(entry["features"]))
        out[group_name] = entry
    return out


def _resolve_symbol(name: str, mapping: Mapping[str, Any], kind: str) -> Any:
    value = mapping.get(name)
    if value is None:
        raise KeyError(f"Unknown {kind}: {name}")
    return value


def _runtime_maps() -> dict[str, dict[str, Any]]:
    return {
        "scaler": {
            "StandardScaler": StandardScaler,
            "DataScaler": DataScaler,
        },
        "dataset": {
            "StyledTransfollowerDataset": StyledTransfollowerDataset,
            "LSTMDataset": LSTMDataset,
        },
        "model": {
            "StyleTransformer": StyleTransformer,
            "Transfollower": Transfollower,
            "CF_LSTM": CF_LSTM,
            "IDM": IDM,
        },
        "activation": {
            "Sigmoid": nn.Sigmoid,
            "Tanh": nn.Tanh,
            "ReLU": nn.ReLU,
        },
        "optimizer": {
            "Adam": optim.Adam,
        },
        "fn": {
            "DEFAULT_PRED_FUNC": DEFAULT_PRED_FUNC,
            "DEFAULT_MASK": DEFAULT_MASK,
            "idm_update_func": idm_update_func,
            "idm_concat": idm_concat,
            "lstm_update_func": lstm_update_func,
            "style_update_func": style_update_func,
            "stylecf_mask": stylecf_mask,
            "transformer_mask": transformer_mask,
            "stack_name": stack_name,
        },
    }


def _build_datahandle_config(raw: dict[str, Any], maps: dict[str, dict[str, Any]]) -> dict[str, Any]:
    data_filter_config = dict(raw["data_filter_config"])
    for key in (
        "acceleration_range",
        "speed_range",
        "spacing_range",
        "dtw_range",
        "thw",
        "pos_tol_range",
        "r_time_range",
        "spd_tol_range",
    ):
        if key in data_filter_config:
            data_filter_config[key] = tuple(data_filter_config[key])

    style_raw = dict(raw["style_data_config"])
    style_data_config = dict(style_raw)
    style_data_config["scaler"] = _resolve_symbol(str(style_raw["scaler"]), maps["scaler"], "scaler")
    style_data_config["dataset"] = _resolve_symbol(str(style_raw["dataset"]), maps["dataset"], "dataset")
    style_data_config["model_name"] = _resolve_symbol(str(style_raw["model_name"]), maps["model"], "model")
    style_data_config["x_groups"] = _resolve_groups(style_raw["x_groups"])
    style_data_config["y_groups"] = _resolve_groups(style_raw["y_groups"])

    lstm_raw = dict(raw["lstm_data_config"])
    lstm_data_config = dict(lstm_raw)
    lstm_data_config["scaler"] = _resolve_symbol(str(lstm_raw["scaler"]), maps["scaler"], "scaler")
    lstm_data_config["dataset"] = _resolve_symbol(str(lstm_raw["dataset"]), maps["dataset"], "dataset")
    lstm_data_config["in_features"] = _resolve_features(list(lstm_raw["in_features"]))
    lstm_data_config["eva_features"] = _resolve_features(list(lstm_raw["eva_features"]))

    return {
        "data_filter_config": data_filter_config,
        "filter_names": list(raw["filter_names"]),
        "style_data_config": style_data_config,
        "lstm_data_config": lstm_data_config,
    }


def _build_train_config(
    raw: dict[str, Any],
    maps: dict[str, dict[str, Any]],
    style_data_config: dict[str, Any],
) -> dict[str, Any]:
    train_raw = dict(raw["style_train_config"])

    device_name = str(train_raw.get("device", "auto"))
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    template = str(train_raw.get("best_model_path_template", "models/best-model-{timestamp}.pth"))
    best_model_path = template.replace("{timestamp}", timestamp)

    loss_name = str(train_raw.get("loss_func", "StyleMultiTaskLoss.default"))
    y_features = style_data_config["y_groups"]["y_seq"]["features"]

    # Backward-compatible alias:
    # "StyleLoss.acc_spacing_mse" -> spacing-only objective.
    if loss_name == "StyleLoss.acc_spacing_mse":
        default_weights = LossWeights(spacing=1.0, acc=0.0, contrastive=0.0)
    else:
        default_weights = LossWeights(
            spacing=float(train_raw.get("w_spacing", 1.0)),
            acc=float(train_raw.get("w_acc", 0.3)),
            contrastive=float(train_raw.get("w_contrastive", 0.1)),
        )

    loss_func = StyleMultiTaskLoss(
        y_features=y_features,
        default_weights=default_weights,
        contrastive_temperature=float(train_raw.get("contrastive_temperature", 0.2)),
    )

    optim_name = str(train_raw.get("optim", "Adam"))
    optim_func = _resolve_symbol(optim_name, maps["optimizer"], "optimizer")

    style_train_config = dict(train_raw)
    style_train_config["device"] = device
    style_train_config["best_model_path"] = best_model_path
    style_train_config["loss_func"] = loss_func
    style_train_config["loss_name"] = loss_name
    style_train_config["optim"] = optim_func
    style_train_config.pop("best_model_path_template", None)

    return {
        "best_model_path": best_model_path,
        "style_train_config": style_train_config,
    }


def _build_models_config(
    raw: dict[str, Any],
    maps: dict[str, dict[str, Any]],
    lstm_data_config: dict[str, Any],
) -> dict[str, Any]:
    lstm_raw = dict(raw["lstm_model_config"])
    lstm_model_config = dict(lstm_raw)
    lstm_model_config["model_name"] = _resolve_symbol(str(lstm_raw["model_name"]), maps["model"], "model")

    num_feature = lstm_raw.get("num_feature", "auto")
    if isinstance(num_feature, str) and num_feature.lower() == "auto":
        num_feature = len(lstm_data_config["in_features"])
    lstm_model_config["num_feature"] = int(num_feature)

    pred_step = lstm_raw.get("pred_step", "auto")
    if isinstance(pred_step, str) and pred_step.lower() == "auto":
        pred_step = int(lstm_data_config["pred_len"])
    lstm_model_config["pred_step"] = int(pred_step)

    activation_name = str(lstm_raw["activation_func"])
    activation_cls = _resolve_symbol(activation_name, maps["activation"], "activation")
    lstm_model_config["activation_func"] = activation_cls()
    if "regular_range" in lstm_model_config:
        lstm_model_config["regular_range"] = tuple(lstm_model_config["regular_range"])

    idm_raw = dict(raw["idm_calibration_config"])
    idm_calibration_config = dict(idm_raw)
    idm_calibration_config["x_groups"] = _resolve_groups(idm_raw["x_groups"])
    idm_calibration_config["y_groups"] = _resolve_groups(idm_raw["y_groups"])

    loss_name = str(idm_raw["loss"])
    if loss_name != "IDMLoss.acc_dis_mse":
        raise ValueError(f"Unsupported IDM loss definition: {loss_name}")
    idm_calibration_config["loss"] = IDMLoss().acc_dis_mse
    idm_calibration_config["update_func"] = _resolve_symbol(
        str(idm_raw["update_func"]),
        maps["fn"],
        "function",
    )
    idm_calibration_config["concat"] = _resolve_symbol(str(idm_raw["concat"]), maps["fn"], "function")
    scaler_name = str(idm_raw["scaler"])
    scaler_cls = _resolve_symbol(scaler_name, maps["scaler"], "scaler")
    idm_calibration_config["scaler"] = scaler_cls()
    idm_calibration_config["pred_func"] = _resolve_symbol(str(idm_raw["pred_func"]), maps["fn"], "function")
    idm_calibration_config["mask"] = _resolve_symbol(str(idm_raw["mask"]), maps["fn"], "function")

    return {
        "lstm_model_config": lstm_model_config,
        "idm_calibration_config": idm_calibration_config,
    }


def _build_test_config(raw: dict[str, Any], maps: dict[str, dict[str, Any]]) -> dict[str, Any]:
    test_raw = dict(raw["test_config"])
    test_config = dict(test_raw)

    for window_key in ("style_window", "test_window", "style_window_before_seconds"):
        if window_key in test_config:
            test_config[window_key] = tuple(test_config[window_key])

    for agent_key in ("style_agent", "lstm_agent", "transformer_agent", "idm_agent"):
        if agent_key not in test_raw:
            continue
        agent_raw = dict(test_raw[agent_key])
        test_config[agent_key] = {
            "model": _resolve_symbol(str(agent_raw["model"]), maps["model"], "model"),
            "pred_func": _resolve_symbol(str(agent_raw["pred_func"]), maps["fn"], "function"),
            "update_func": _resolve_symbol(str(agent_raw["update_func"]), maps["fn"], "function"),
            "mask": _resolve_symbol(str(agent_raw["mask"]), maps["fn"], "function"),
            "concat_func": _resolve_symbol(str(agent_raw["concat_func"]), maps["fn"], "function"),
        }

    return {"test_config": test_config}


def _build_bundle() -> dict[str, Any]:
    maps = _runtime_maps()
    data_raw = _load_toml("datahandle.toml")
    train_raw = _load_toml("train.toml")
    model_raw = _load_toml("models.toml")
    test_raw = _load_test_toml()

    data_cfg = _build_datahandle_config(data_raw, maps)
    train_cfg = _build_train_config(train_raw, maps, data_cfg["style_data_config"])
    model_cfg = _build_models_config(model_raw, maps, data_cfg["lstm_data_config"])
    test_cfg = _build_test_config(test_raw, maps)

    test_config = test_cfg["test_config"]
    style_window = tuple(test_config.get("style_window", (0, 300)))
    test_window = tuple(test_config.get("test_window", (300, 900)))

    return {
        **data_cfg,
        **train_cfg,
        **model_cfg,
        **test_cfg,
        "DEFAULT_STYLE_WINDOW": style_window,
        "DEFAULT_TEST_WINDOW": test_window,
    }


def get_exps_configs(force_reload: bool = False) -> dict[str, Any]:
    global _CACHE
    if _CACHE is None or force_reload:
        _CACHE = _build_bundle()
    return _CACHE


def get_exps_config(name: str, force_reload: bool = False) -> Any:
    cfg = get_exps_configs(force_reload=force_reload)
    if name not in cfg:
        raise KeyError(f"Unknown config key: {name}")
    return cfg[name]
