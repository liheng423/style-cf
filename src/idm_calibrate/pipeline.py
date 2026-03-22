from __future__ import annotations

import os

from .calibrator import calibrate_idm
from .config import idm_calibration_config
from ..exps.datahandle.config import data_filter_config, filter_names
from ..exps.models.idm import IDM
from ..exps.train.model_trainer import build_dataset
from ..exps.utils.utils import build_id_datapack, load_zen_data


def _resolve_data_path(data_path: str | None = None) -> str:
    env_path = os.environ.get("ZEN_DATA_PATH")
    path = data_path or env_path
    if not path:
        raise ValueError("Dataset path is missing. Pass data_path or set ZEN_DATA_PATH.")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")
    return path


def _dataset(head: int | None = None, data_path: str | None = None):
    d = load_zen_data(_resolve_data_path(data_path), rise=True, in_kph=False, kilo_norm=True)
    return d.head(head) if head is not None else d


def run_calibration(
    head: int | None = 10000,
    data_path: str | None = None,
    config: dict | None = None,
):
    datapack = _dataset(head=head, data_path=data_path)
    datapack = build_dataset(datapack, filter_names, data_filter_config)
    id_datapack = build_id_datapack(datapack, require_const_self_id=True, key_by_id=False)

    resolved_config = dict(idm_calibration_config if config is None else config)
    resolved_config.setdefault("sample_size", 1000)

    return calibrate_idm(IDM, id_datapack, resolved_config)


def main() -> int:
    run_calibration()
    return 0


__all__ = ["run_calibration", "main"]
