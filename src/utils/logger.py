from __future__ import annotations

import logging
from typing import Any


def _build_default_logger() -> logging.Logger:
    logger = logging.getLogger("stylecf")
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


logger = _build_default_logger()


def get_with_warn(payload: dict, key: str, default) -> Any:
    if key not in payload or payload[key] is None:
        logger.warning(f"x_payload missing '{key}', using default value")
        return default
    return payload[key]
