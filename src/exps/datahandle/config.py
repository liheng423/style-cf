from __future__ import annotations

from ...utils.config_utils import get_exps_configs

_CFG = get_exps_configs()

data_filter_config = _CFG["data_filter_config"]
filter_names = _CFG["filter_names"]
style_data_config = _CFG["style_data_config"]
lstm_data_config = _CFG["lstm_data_config"]
