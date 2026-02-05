
import time
from functools import wraps
from typing import Any
from src.exps.configs import style_train_config
from logging import getLogger

logger = getLogger(__name__)


def get_with_warn(payload: dict, key: str, default) -> Any:
    if key not in payload or payload[key] is None:
        logger.warning(f"x_payload missing '{key}', using default value")
        return default
    return payload[key]

