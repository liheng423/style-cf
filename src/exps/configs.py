from __future__ import annotations

from ..utils.config_utils import get_exps_configs

_CFG = get_exps_configs()

best_model_path = _CFG["best_model_path"]
data_filter_config = _CFG["data_filter_config"]
filter_names = _CFG["filter_names"]
idm_calibration_config = _CFG["idm_calibration_config"]
lstm_data_config = _CFG["lstm_data_config"]
lstm_model_config = _CFG["lstm_model_config"]
style_data_config = _CFG["style_data_config"]
style_train_config = _CFG["style_train_config"]
test_config = _CFG["test_config"]
DEFAULT_STYLE_WINDOW = _CFG["DEFAULT_STYLE_WINDOW"]
DEFAULT_TEST_WINDOW = _CFG["DEFAULT_TEST_WINDOW"]


__all__ = [
    "best_model_path",
    "data_filter_config",
    "filter_names",
    "idm_calibration_config",
    "lstm_data_config",
    "lstm_model_config",
    "style_data_config",
    "style_train_config",
    "test_config",
    "DEFAULT_STYLE_WINDOW",
    "DEFAULT_TEST_WINDOW",
]
