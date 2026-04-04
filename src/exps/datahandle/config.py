from __future__ import annotations

from ...utils.config_utils import get_datahandle_config, get_models_config

_DATA_CFG = get_datahandle_config()
_MODEL_CFG = get_models_config()

data_filter_config = _DATA_CFG["data_filter_config"]
filter_names = _DATA_CFG["filter_names"]
lstm_data_config = _MODEL_CFG["lstm_data_config"]
lstm_dataset_config = _MODEL_CFG["lstm_dataset_config"]
lstm_model_io_config = _MODEL_CFG["lstm_model_io_config"]
lstm_runtime_components = _MODEL_CFG["lstm_runtime_components"]
style_dataset_config = _MODEL_CFG["style_dataset_config"]
style_data_config = _MODEL_CFG["style_data_config"]
style_model_io_config = _MODEL_CFG["style_model_io_config"]
style_runtime_components = _MODEL_CFG["style_runtime_components"]
