from __future__ import annotations

from ..utils.config_utils import get_exps_configs

_CFG = get_exps_configs()

idm_calibration_config = _CFG["idm_calibration_config"]

__all__ = ["idm_calibration_config"]
