from __future__ import annotations

from ...utils.config_utils import get_exps_configs

_CFG = get_exps_configs()

best_model_path = _CFG["best_model_path"]
style_train_config = _CFG["style_train_config"]
