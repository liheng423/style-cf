from __future__ import annotations

from ..utils.config_utils import (
    get_datahandle_config,
    get_idm_calibration_config,
    get_models_config,
    get_test_configs,
    get_train_configs,
)

_DATA_CFG = get_datahandle_config()
_MODEL_CFG = get_models_config()
_TRAIN_CFG = get_train_configs()
_TEST_CFG = get_test_configs()
_IDM_CFG = get_idm_calibration_config()

data_filter_config = _DATA_CFG["data_filter_config"]
filter_names = _DATA_CFG["filter_names"]
idm_calibration_config = _IDM_CFG["idm_calibration_config"]
lstm_data_config = _MODEL_CFG["lstm_data_config"]
lstm_dataset_config = _MODEL_CFG["lstm_dataset_config"]
lstm_model_io_config = _MODEL_CFG["lstm_model_io_config"]
lstm_model_config = _MODEL_CFG["lstm_model_config"]
lstm_runtime_components = _MODEL_CFG["lstm_runtime_components"]
style_dataset_config = _MODEL_CFG["style_dataset_config"]
style_data_config = _MODEL_CFG["style_data_config"]
style_model_io_config = _MODEL_CFG["style_model_io_config"]
style_runtime_components = _MODEL_CFG["style_runtime_components"]
style_train_config = _TRAIN_CFG["style_train_config"]
transformer_train_config = _TRAIN_CFG["transformer_train_config"]
lstm_train_config = _TRAIN_CFG["lstm_train_config"]
train_entry_config = _TRAIN_CFG["train_entry_config"]
train_config_meta = _TRAIN_CFG["train_config_meta"]
test_config = _TEST_CFG["test_config"]
DEFAULT_STYLE_WINDOW = _TEST_CFG["DEFAULT_STYLE_WINDOW"]
DEFAULT_TEST_WINDOW = _TEST_CFG["DEFAULT_TEST_WINDOW"]


__all__ = [
    "data_filter_config",
    "filter_names",
    "idm_calibration_config",
    "lstm_data_config",
    "lstm_dataset_config",
    "lstm_model_io_config",
    "lstm_model_config",
    "lstm_runtime_components",
    "style_dataset_config",
    "style_data_config",
    "style_model_io_config",
    "style_runtime_components",
    "style_train_config",
    "transformer_train_config",
    "lstm_train_config",
    "train_entry_config",
    "train_config_meta",
    "test_config",
    "DEFAULT_STYLE_WINDOW",
    "DEFAULT_TEST_WINDOW",
]
