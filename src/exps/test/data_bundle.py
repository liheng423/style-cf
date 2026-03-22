from __future__ import annotations

from typing import Any

import numpy as np

from ...schema import CFNAMES as CF
from ..datahandle.databuilder import build_dataset
from ..datahandle.feat_extractor import batch_apply, reaction_time, time_headway


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


__all__ = ["build_style_datapack", "split_eval_windows"]
