from __future__ import annotations

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Mapping

try:
    import tomllib  # type: ignore[attr-defined]
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[no-redef]

from ..schema import CFNAMES as CF


_CONFIG_ROOT = Path(__file__).resolve().parents[1] / "exps" / "configs"
_DEFAULT_CONFIG_DIR = _CONFIG_ROOT / "default_configs"
_TRAINING_CONFIG_DIR = _CONFIG_ROOT / "training"
_TESTING_CONFIG_DIR = _CONFIG_ROOT / "testing"
_CACHE: dict[str, dict[str, Any]] = {}


def _load_toml_path(path: Path) -> dict[str, Any]:
    """Load one TOML file from disk."""
    with path.open("rb") as f:
        return tomllib.load(f)


def _load_default_toml(name: str) -> dict[str, Any]:
    """Load one config from `default_configs/` by filename."""
    return _load_toml_path(_DEFAULT_CONFIG_DIR / name)


def _deep_merge_dict(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge nested mappings while keeping scalar override semantics."""
    merged: dict[str, Any] = dict(base)
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            merged[key] = _deep_merge_dict(current, value)
        else:
            merged[key] = value
    return merged


def _load_active_profile(
    profile_dir: Path,
    *,
    section: str,
    base_payload: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load one active profile and merge it on top of a given default payload."""
    active_path = profile_dir / "active.toml"
    meta = {
        "profile_name": "<default>",
        "profile_path": "",
        "active_path": str(active_path),
    }
    if not active_path.exists():
        return dict(base_payload), meta

    try:
        active_payload = _load_toml_path(active_path)
    except Exception:
        return dict(base_payload), meta

    section_payload = active_payload.get(section)
    if not isinstance(section_payload, Mapping):
        return dict(base_payload), meta

    active_name = section_payload.get("active")
    if not isinstance(active_name, str) or not active_name.strip():
        return dict(base_payload), meta

    filename = active_name.strip()
    if not filename.lower().endswith(".toml"):
        filename = f"{filename}.toml"
    filename = Path(filename).name

    profile_path = profile_dir / filename
    if not profile_path.exists():
        stem = Path(filename).stem.lower()
        candidates = [p for p in profile_dir.glob("*.toml") if p.name.lower() != "active.toml"]
        relaxed = [p for p in candidates if stem in p.stem.lower() or p.stem.lower() in stem]
        if len(relaxed) == 1:
            profile_path = relaxed[0]
        else:
            return dict(base_payload), meta

    merged = _deep_merge_dict(base_payload, _load_toml_path(profile_path))
    meta["profile_name"] = profile_path.name
    meta["profile_path"] = str(profile_path)
    return merged, meta


def _feature_name(name: str) -> str:
    """Resolve a symbolic feature name to the canonical schema constant when available."""
    return getattr(CF, name, name)


def _resolve_features(features: list[str]) -> list[str]:
    """Resolve a flat feature list from config text to runtime feature keys."""
    return [_feature_name(name) for name in features]


def _resolve_groups(groups: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Resolve every `features` field inside grouped feature config blocks."""
    out: dict[str, dict[str, Any]] = {}
    for group_name, group in groups.items():
        entry = dict(group)
        if "features" in entry:
            entry["features"] = _resolve_features(list(entry["features"]))
        out[group_name] = entry
    return out


def _resolve_symbol(name: str, mapping: Mapping[str, Any], kind: str) -> Any:
    """Resolve a configured symbolic name to a concrete runtime object."""
    value = mapping.get(name)
    if value is None:
        raise KeyError(f"Unknown {kind}: {name}")
    return value


def _require_mapping_entry(raw: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    """Read one required mapping entry from a config payload."""
    value = raw.get(key)
    if not isinstance(value, Mapping):
        raise KeyError(f"Missing required mapping config: '{key}'")
    return value


def _require_list_entry(raw: Mapping[str, Any], key: str) -> list[Any]:
    """Read one required list entry from a config payload."""
    value = raw.get(key)
    if not isinstance(value, list):
        raise KeyError(f"Missing required list config: '{key}'")
    return value


def _require_nested_mapping(raw: Mapping[str, Any], *keys: str) -> Mapping[str, Any]:
    """Read one required nested mapping entry from a config payload."""
    current: Mapping[str, Any] = raw
    walked: list[str] = []
    for key in keys:
        walked.append(key)
        value = current.get(key)
        if not isinstance(value, Mapping):
            raise KeyError(f"Missing required mapping config: '{'.'.join(walked)}'")
        current = value
    return current


def _runtime_maps() -> dict[str, dict[str, Any]]:
    """Central registry for converting config strings into runtime classes/functions."""
    import torch.nn as nn
    import torch.optim as optim
    from sklearn.preprocessing import StandardScaler

    from ..exps.datahandle.datascalers import DataScaler
    from ..exps.datahandle.datasets import LSTMDataset, StyledTransfollowerDataset
    from ..exps.losses import IDMLoss
    from ..exps.models.idm import DEFAULT_MASK, DEFAULT_PRED_FUNC, IDM, idm_concat, idm_update_func
    from ..exps.models.lstm import CF_LSTM, lstm_update_func
    from ..exps.models.stylecf import StyleTransformer, style_update_func, stylecf_mask, transformer_mask
    from ..exps.models.transfollower import Transfollower
    from ..exps.utils.utils import stack_name

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
        "loss": {
            "IDMLoss.acc_dis_mse": IDMLoss().acc_dis_mse,
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


def resolve_common_runtime(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve the raw common payload into runtime-ready shared config objects."""
    maps = _runtime_maps()
    return {
        **build_datahandle_config(dict(raw), maps),
        **build_models_config(dict(raw), maps, None),
    }


def build_datahandle_config(raw: dict[str, Any], maps: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    """Build the shared data filtering config used by train/test/calibration entrypoints."""
    _ = maps
    data_filter_config = dict(_require_mapping_entry(raw, "data_filter_config"))
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

    return {
        "data_filter_config": data_filter_config,
        "filter_names": list(_require_list_entry(raw, "filter_names")),
    }


def build_models_config(
    raw: dict[str, Any],
    maps: dict[str, dict[str, Any]],
    lstm_data_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build model-adjacent configs shared across train/test/style/idm entrypoints."""
    _ = lstm_data_config
    general_config = dict(_require_mapping_entry(raw, "general"))
    style_dataset_config = dict(_require_nested_mapping(raw, "stylecf", "dataset"))

    lstm_raw = dict(_require_mapping_entry(raw, "lstm_data_config"))
    lstm_model_io_config = dict(_require_mapping_entry(lstm_raw, "model_io"))
    lstm_dataset_config = dict(_require_mapping_entry(lstm_raw, "dataset_config"))
    lstm_runtime_components = dict(_require_mapping_entry(lstm_raw, "runtime_components"))
    lstm_x_groups = _resolve_groups(dict(_require_mapping_entry(lstm_raw, "x_groups")))
    lstm_y_groups = _resolve_groups(dict(_require_mapping_entry(lstm_raw, "y_groups")))
    lstm_config = {
        **general_config,
        **lstm_model_io_config,
        **lstm_dataset_config,
        **lstm_runtime_components,
        "sample_dt": float(general_config["dt"]),
        "x_groups": lstm_x_groups,
        "y_groups": lstm_y_groups,
        "in_features": list(_require_nested_mapping(lstm_x_groups, "inputs")["features"]),
        "eva_features": list(_require_nested_mapping(lstm_y_groups, "eval")["features"]),
    }
    if "scaler" in lstm_runtime_components:
        lstm_config["scaler"] = _resolve_symbol(str(lstm_runtime_components["scaler"]), maps["scaler"], "scaler")
    if "dataset" in lstm_runtime_components:
        lstm_config["dataset"] = _resolve_symbol(str(lstm_runtime_components["dataset"]), maps["dataset"], "dataset")

    style_raw = dict(_require_mapping_entry(raw, "style_data_config"))
    style_model_io_config = dict(_require_mapping_entry(style_raw, "model_io"))
    style_runtime_raw = dict(_require_mapping_entry(style_raw, "runtime_components"))
    style_runtime_components = {
        "scaler": _resolve_symbol(str(style_runtime_raw["scaler"]), maps["scaler"], "scaler"),
        "dataset": _resolve_symbol(str(style_runtime_raw["dataset"]), maps["dataset"], "dataset"),
        "model_name": _resolve_symbol(str(style_runtime_raw["model_name"]), maps["model"], "model"),
    }
    style_data_config = {
        **general_config,
        **style_dataset_config,
        **style_model_io_config,
        **style_runtime_components,
        "sample_dt": float(general_config["dt"]),
        "x_groups": _resolve_groups(dict(_require_mapping_entry(style_raw, "x_groups"))),
        "y_groups": _resolve_groups(dict(_require_mapping_entry(style_raw, "y_groups"))),
    }

    idm_config = dict(_require_mapping_entry(raw, "idm_calibration_config"))
    if "x_groups" in idm_config and isinstance(idm_config["x_groups"], Mapping):
        idm_config["x_groups"] = _resolve_groups(dict(idm_config["x_groups"]))
    if "y_groups" in idm_config and isinstance(idm_config["y_groups"], Mapping):
        idm_config["y_groups"] = _resolve_groups(dict(idm_config["y_groups"]))

    return {
        "general_config": general_config,
        "style_dataset_config": style_dataset_config,
        "style_model_io_config": style_model_io_config,
        "style_runtime_components": style_runtime_components,
        "style_data_config": style_data_config,
        "lstm_data_config": lstm_config,
        "lstm_model_io_config": lstm_model_io_config,
        "lstm_dataset_config": lstm_dataset_config,
        "lstm_runtime_components": {
            **lstm_runtime_components,
            "scaler": lstm_config["scaler"],
            "dataset": lstm_config["dataset"],
        },
        "lstm_model_config": dict(_require_mapping_entry(raw, "lstm_arch_config")),
        "idm_calibration_config": idm_config,
    }


def _build_style_loss(train_cfg: Mapping[str, Any], style_data_config: Mapping[str, Any]) -> StyleMultiTaskLoss:
    """Convert the style loss declaration into the concrete runtime loss object."""
    from ..exps.losses import LossWeights, StyleMultiTaskLoss

    loss_name = str(train_cfg["loss_func"])
    y_features = style_data_config["y_groups"]["y_seq"]["features"]
    if loss_name == "StyleLoss.acc_spacing_mse":
        default_weights = LossWeights(spacing=1.0, acc=0.0, contrastive=0.0)
    elif loss_name == "StyleMultiTaskLoss.default":
        default_weights = LossWeights(
            spacing=float(_require_option(train_cfg, "w_spacing")),
            acc=float(_require_option(train_cfg, "w_acc")),
            contrastive=float(_require_option(train_cfg, "w_contrastive")),
        )
    else:
        raise ValueError(f"Unsupported style loss definition: {loss_name}")

    return StyleMultiTaskLoss(
        y_features=y_features,
        default_weights=default_weights,
        contrastive_temperature=float(_require_option(train_cfg, "contrastive_temperature")),
    )


def _resolve_train_runtime(
    raw_cfg: Mapping[str, Any],
    maps: dict[str, dict[str, Any]],
    style_data_config: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Apply runtime-only train config normalization after profile/default merge."""
    cfg = dict(raw_cfg)
    if "split_idx_path" in cfg and "split_index_path" not in cfg:
        cfg["split_index_path"] = cfg.pop("split_idx_path")
    if "optim" in cfg:
        cfg["optim"] = _resolve_symbol(str(cfg["optim"]), maps["optimizer"], "optimizer")
    if "best_model_path_template" in cfg:
        template = str(cfg["best_model_path_template"])
        cfg["best_model_path"] = template.replace("{timestamp}", datetime.now().strftime("%Y%m%d-%H%M%S"))
    if style_data_config is not None and "loss_func" in cfg:
        cfg["loss_func"] = _build_style_loss(cfg, style_data_config)
    return cfg


def build_train_config(
    raw: dict[str, Any],
    maps: dict[str, dict[str, Any]],
    style_data_config: dict[str, Any],
) -> dict[str, Any]:
    """Build all training entry configs instead of one monolithic bundle."""
    train_base_config = dict(_require_mapping_entry(raw, "train_base_config"))
    out: dict[str, Any] = {
        "train_entry_config": dict(_require_mapping_entry(raw, "train_entry_config")),
        "train_config_meta": dict(_require_mapping_entry(raw, "train_config_meta")),
    }

    for cfg_name in ("style_train_config", "transformer_train_config", "lstm_train_config"):
        cfg_raw = raw.get(cfg_name, {})
        cfg_merged = _deep_merge_dict(train_base_config, cfg_raw if isinstance(cfg_raw, Mapping) else {})
        out[cfg_name] = _resolve_train_runtime(
            cfg_merged,
            maps,
            style_data_config if cfg_name == "style_train_config" else None,
        )

    return out


def _flatten_test_sections(test_config: Mapping[str, Any]) -> dict[str, Any]:
    """Flatten nested testing sections so the legacy testing code can consume one dict."""
    out = dict(test_config)
    for section_name in ("style_token", "eval_time"):
        section = out.pop(section_name, None)
        if isinstance(section, Mapping):
            out.update(section)
    return out


def build_test_config(raw: dict[str, Any], maps: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Build the testing entry config and resolve model/function references eagerly."""
    test_config = _flatten_test_sections(dict(_require_mapping_entry(raw, "test_config")))

    for window_key in ("style_window", "test_window", "style_window_before_seconds"):
        if window_key in test_config:
            test_config[window_key] = tuple(test_config[window_key])

    for agent_key in ("style_agent", "lstm_agent", "transformer_agent", "idm_agent"):
        raw_agent = raw.get(agent_key)
        if raw_agent is None:
            continue
        agent_config = dict(raw_agent)
        if "model" in agent_config:
            agent_config["model"] = _resolve_symbol(str(agent_config["model"]), maps["model"], "model")
        if "pred_func" in agent_config:
            agent_config["pred_func"] = _resolve_symbol(str(agent_config["pred_func"]), maps["fn"], "function")
        if "update_func" in agent_config:
            agent_config["update_func"] = _resolve_symbol(str(agent_config["update_func"]), maps["fn"], "function")
        if "mask" in agent_config:
            agent_config["mask"] = _resolve_symbol(str(agent_config["mask"]), maps["fn"], "function")
        if "concat_func" in agent_config:
            agent_config["concat_func"] = _resolve_symbol(str(agent_config["concat_func"]), maps["fn"], "function")
        test_config[agent_key] = agent_config

    style_window = tuple(_require_option(test_config, "style_window"))
    test_window = tuple(_require_option(test_config, "test_window"))
    return {
        "test_config": test_config,
        "DEFAULT_STYLE_WINDOW": style_window,
        "DEFAULT_TEST_WINDOW": test_window,
    }


def _build_style_pip_config(raw: dict[str, Any]) -> dict[str, Any]:
    """Build the standalone style-analysis pipeline config."""
    import torch

    if "style_pipeline_config" in raw:
        style_pipeline_raw = raw["style_pipeline_config"]
    elif "test_config" in raw and isinstance(raw["test_config"], Mapping):
        style_pipeline_raw = raw["test_config"].get("style_pipeline")
    else:
        style_pipeline_raw = raw

    if not isinstance(style_pipeline_raw, Mapping):
        raise TypeError("style pipeline config must be a mapping.")

    style_pipeline_config = dict(style_pipeline_raw)
    if "device" in style_pipeline_config:
        style_pipeline_config["device"] = torch.device(str(style_pipeline_config["device"]))
    if "output_dir" in style_pipeline_config:
        style_pipeline_config["output_dir"] = Path(str(style_pipeline_config["output_dir"]))
    if "cluster" in style_pipeline_config:
        cluster_raw = style_pipeline_config["cluster"]
        if not isinstance(cluster_raw, Mapping):
            raise TypeError("style_pipeline_config['cluster'] must be a mapping.")
        cluster_protocol = dict(cluster_raw)
        cluster_protocol["method"] = str(cluster_protocol["method"]).strip().lower()
        style_pipeline_config["cluster"] = cluster_protocol
    return {"style_pipeline_config": style_pipeline_config}


def build_style_config(raw: dict[str, Any], maps: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Public builder for the style-analysis entrypoint."""
    _ = maps
    return _build_style_pip_config(raw)


def build_datasplit_config(raw: dict[str, Any], maps: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    """Build the dedicated data-splitting entry config."""
    _ = maps
    return {"data_split_config": dict(_require_mapping_entry(raw, "data_split_config"))}


def build_idm_calibration_config(raw: dict[str, Any], maps: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Build the IDM calibration entry config and resolve runtime callables."""
    import torch

    cfg = dict(_require_mapping_entry(raw, "idm_calibration_config"))
    if "x_groups" in cfg and isinstance(cfg["x_groups"], Mapping):
        cfg["x_groups"] = _resolve_groups(dict(cfg["x_groups"]))
    if "y_groups" in cfg and isinstance(cfg["y_groups"], Mapping):
        cfg["y_groups"] = _resolve_groups(dict(cfg["y_groups"]))
    if "loss" in cfg:
        cfg["loss"] = _resolve_symbol(str(cfg["loss"]), maps["loss"], "loss")
    for key, mapping_name in (
        ("update_func", "fn"),
        ("concat", "fn"),
        ("pred_func", "fn"),
        ("mask", "fn"),
        ("scaler", "scaler"),
    ):
        if key in cfg:
            cfg[key] = _resolve_symbol(str(cfg[key]), maps[mapping_name], mapping_name)
    if "device" in cfg:
        cfg["device"] = torch.device(str(cfg["device"]))
    return {"idm_calibration_config": cfg}


def _get_cached(name: str, builder: Callable[[], dict[str, Any]], force_reload: bool = False) -> dict[str, Any]:
    """Memoize one config family so repeated imports do not keep reparsing TOML."""
    if force_reload or name not in _CACHE:
        _CACHE[name] = builder()
    return _CACHE[name]


def load_common_config(force_reload: bool = False) -> dict[str, Any]:
    """Load the raw common payload shared by train/test/style without runtime resolution."""
    _ = force_reload
    raw = _load_default_toml("datahandle.toml")
    raw = _deep_merge_dict(raw, _load_default_toml("models.toml"))
    raw = _deep_merge_dict(raw, _load_default_toml("train_datasets.toml"))
    raw = _deep_merge_dict(raw, _load_default_toml("general_cfg.toml"))
    return {"config": raw}




def get_common_configs(force_reload: bool = False) -> dict[str, Any]:
    """Return cached common runtime config used across multiple entrypoints."""
    return _get_cached(
        "common",
        lambda: resolve_common_runtime(load_common_config(force_reload=force_reload)["config"]),
        force_reload=force_reload,
    )


def get_datahandle_config(force_reload: bool = False) -> dict[str, Any]:
    """Return only the shared data filtering config."""
    return _get_cached(
        "datahandle",
        lambda: build_datahandle_config(_load_default_toml("datahandle.toml"), None),
        force_reload=force_reload,
    )


def get_models_config(force_reload: bool = False) -> dict[str, Any]:
    """Return only the shared model/data-shape config family."""
    def _builder() -> dict[str, Any]:
        raw = _load_default_toml("models.toml")
        raw = _deep_merge_dict(raw, _load_default_toml("train_datasets.toml"))
        raw = _deep_merge_dict(raw, _load_default_toml("general_cfg.toml"))
        return build_models_config(raw, _runtime_maps(), None)

    return _get_cached("models", _builder, force_reload=force_reload)


def get_train_configs(force_reload: bool = False) -> dict[str, Any]:
    """Return the cached training config family after active-profile merge."""
    def _builder() -> dict[str, Any]:
        raw, meta = _load_active_profile(
            _TRAINING_CONFIG_DIR,
            section="training",
            base_payload=_load_default_toml("train.toml"),
        )
        raw["train_config_meta"] = meta
        return build_train_config(raw, _runtime_maps(), get_models_config(force_reload=force_reload)["style_data_config"])

    return _get_cached("train", _builder, force_reload=force_reload)


def get_test_configs(force_reload: bool = False) -> dict[str, Any]:
    """Return the cached testing config family after active-profile merge."""
    def _builder() -> dict[str, Any]:
        raw, _ = _load_active_profile(
            _TESTING_CONFIG_DIR,
            section="testing",
            base_payload=_load_default_toml("test.toml"),
        )
        model_raw = _load_default_toml("models.toml")
        for agent_key in ("style_agent", "lstm_agent", "transformer_agent", "idm_agent"):
            if agent_key in model_raw:
                raw[agent_key] = model_raw[agent_key]
        return build_test_config(raw, _runtime_maps())

    return _get_cached("test", _builder, force_reload=force_reload)


def load_style_pipeline_config(force_reload: bool = False) -> dict[str, Any]:
    """Load the raw standalone style-analysis payload before runtime resolution."""
    def _builder() -> dict[str, Any]:
        raw = _load_default_toml("style.toml")
        testing_raw, _ = _load_active_profile(
            _TESTING_CONFIG_DIR,
            section="testing",
            base_payload={},
        )
        if testing_raw:
            raw = _deep_merge_dict(raw, testing_raw)
        return {"config": raw}

    return _get_cached("style_raw", _builder, force_reload=force_reload)


def get_style_configs(force_reload: bool = False) -> dict[str, Any]:
    """Return the cached runtime config for the style-analysis entrypoint."""
    return _get_cached(
        "style",
        lambda: build_style_config(dict(load_style_pipeline_config(force_reload=force_reload)["config"]), _runtime_maps()),
        force_reload=force_reload,
    )


def get_datasplit_configs(force_reload: bool = False) -> dict[str, Any]:
    """Return the cached config for the dedicated data-splitting entrypoint."""
    return _get_cached(
        "datasplit",
        lambda: build_datasplit_config(_load_default_toml("split_data.toml"), None),
        force_reload=force_reload,
    )


def get_idm_calibration_config(force_reload: bool = False) -> dict[str, Any]:
    """Return the cached config for the IDM calibration entrypoint."""
    def _builder() -> dict[str, Any]:
        raw = _load_default_toml("idm_calibration.toml")
        raw = _deep_merge_dict(
            raw,
            {"idm_calibration_config": get_models_config(force_reload=force_reload)["idm_calibration_config"]},
        )
        return build_idm_calibration_config(raw, _runtime_maps())

    return _get_cached("idm_calibration", _builder, force_reload=force_reload)


def get_exps_configs(force_reload: bool = False) -> dict[str, Any]:
    """Backward-compatible aggregate accessor kept only for older import sites."""
    return {
        **get_common_configs(force_reload=force_reload),
        **get_train_configs(force_reload=force_reload),
        **get_test_configs(force_reload=force_reload),
        **get_idm_calibration_config(force_reload=force_reload),
    }


def _require_option(opt: Mapping[str, Any] | SimpleNamespace, name: str):
    """Read one required option from either a mapping or a namespace object."""
    if isinstance(opt, Mapping):
        if name not in opt:
            raise ValueError(f"Missing '{name}'.")
        return opt[name]
    if not hasattr(opt, name):
        raise ValueError(f"Missing '{name}'.")
    return getattr(opt, name)


__all__ = [
    "build_datahandle_config",
    "build_models_config",
    "build_train_config",
    "build_test_config",
    "build_style_config",
    "build_datasplit_config",
    "build_idm_calibration_config",
    "load_common_config",
    "resolve_common_runtime",
    "load_style_pipeline_config",
    "get_common_configs",
    "get_datahandle_config",
    "get_models_config",
    "get_train_configs",
    "get_test_configs",
    "get_style_configs",
    "get_datasplit_configs",
    "get_idm_calibration_config",
    "get_exps_configs",
    "_require_option",
]
