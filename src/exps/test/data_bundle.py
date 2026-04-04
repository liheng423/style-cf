from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ...schema import CFNAMES as CF
from ...utils.rawdata_loader import load_datapack
from ..datahandle.datapackbuilder import build_dataset
from ..datahandle.feat_extractor import batch_apply, reaction_time, time_headway
from .scaler_config import load_test_scalers
from ..utils.split_io import load_split_indices


@dataclass(frozen=True)
class EvalBundle:
    d_style: Any
    d_test: Any
    style_scalers: dict[str, object]
    transformer_scalers: dict[str, object]
    lstm_scalers: dict[str, object]


def build_style_datapack(raw_data: Any, filter_names: list[Any], data_filter_config: dict[str, Any]):
    d_full = build_dataset(raw_data, filter_names, data_filter_config)

    num_samples = d_full.data.shape[0]
    num_steps = d_full.data.shape[1]
    time_axis = np.arange(num_steps, dtype=np.float32) * float(d_full.dt)

    d_full.append_col(
        time_axis.reshape(1, -1, 1).repeat(num_samples, axis=0),
        CF.TIME,
    )
    d_full.append_col(
        batch_apply(
            reaction_time,
            [d_full[:, :, CF.LEAD_V], d_full[:, :, CF.SELF_V], d_full[:, :, CF.TIME]],
        )[:, :, None],
        CF.REACT,
    )
    d_full.append_col(
        batch_apply(
            time_headway,
            [d_full[:, :, CF.DELTA_X] - d_full[:, :, CF.LEAD_L], d_full[:, :, CF.SELF_V]],
        ),
        CF.THW,
    )
    return d_full


def split_eval_windows(d_full: Any, style_window: tuple[int, int], test_window: tuple[int, int]):
    total_steps = d_full.data.shape[1]
    for name, window in (("style_window", style_window), ("test_window", test_window)):
        start, end = window
        if not (0 <= start < end <= total_steps):
            raise ValueError(
                f"{name}={window} is invalid for sequence length {total_steps}. "
                "Update test_config['style_window'/'test_window']."
            )
    return d_full.split_by_time_windows_list([style_window, test_window])


def _select_partition(d_full: Any, split_index_path: str | None, split_partition: str):
    partition = str(split_partition).strip().lower()
    if partition in {"", "all"}:
        return d_full
    if not split_index_path:
        raise ValueError(f"split_index_path is required when split_partition='{partition}'.")

    payload = load_split_indices(Path(str(split_index_path)))
    key_map = {
        "train": "train_idx",
        "val": "val_idx",
        "valid": "val_idx",
        "validation": "val_idx",
        "test": "test_idx",
    }
    idx_key = key_map.get(partition)
    if idx_key is None:
        raise ValueError(
            f"Unsupported split_partition='{split_partition}'. Expected one of: all, train, val, test."
        )

    idx = np.asarray(payload[idx_key], dtype=np.int64)
    return d_full.select_rows(idx)


def load_eval_bundle(
    test_config: Mapping[str, Any],
    style_data_config: Mapping[str, Any],
    lstm_data_config: Mapping[str, Any],
    filter_names: list[Any],
    data_filter_config: dict[str, Any],
    head: int | None = None,
    use_split_windows: bool = True,
    use_full_window: bool = False,
    split_log_prefix: str = "Eval",
) -> EvalBundle:
    rawdata_name = test_config.get("rawdata_config", test_config.get("datacfg"))
    if rawdata_name in (None, "", ...):
        raise ValueError(
            "Missing rawdata config for eval bundle. "
            "Set style_pipeline_config['rawdata_config']."
        )

    datapack, _, _ = load_datapack(str(rawdata_name))
    if head is not None:
        datapack = datapack.head(int(head))

    d_full = build_style_datapack(datapack, filter_names, data_filter_config)
    split_partition = str(test_config.get("split_partition", "all"))
    split_index_path = test_config.get("split_index_path")
    d_full = _select_partition(
        d_full,
        None if split_index_path in (None, "", ...) else str(split_index_path),
        split_partition,
    )

    if use_full_window:
        d_style = d_full
        d_test = d_full
    elif use_split_windows:
        style_window = tuple(test_config.get("style_window", (0, int(d_full.data.shape[1]))))
        test_window = tuple(test_config.get("test_window", style_window))
        d_style, d_test = split_eval_windows(d_full, style_window, test_window)
    else:
        raise ValueError("Either use_split_windows or use_full_window must be True for eval bundle.")

    style_scalers, transformer_scalers, lstm_scalers = load_test_scalers(
        test_config,
        style_data_config["x_groups"],
    )

    return EvalBundle(
        d_style=d_style,
        d_test=d_test,
        style_scalers=style_scalers,
        transformer_scalers=transformer_scalers,
        lstm_scalers=lstm_scalers,
    )


__all__ = ["EvalBundle", "build_style_datapack", "split_eval_windows", "load_eval_bundle"]
