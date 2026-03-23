from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter

from .regime import ACC_REGIME_IDX, DEC_REGIME_IDX, map_segment_to_regime_index


def _time_axis(x: float | np.ndarray | list[float], length: int) -> np.ndarray:
    if isinstance(x, (list, np.ndarray)):
        arr = np.asarray(x, dtype=float)
        if arr.shape[0] != length:
            raise ValueError(f"time axis length mismatch: expected {length}, got {arr.shape[0]}")
        return arr
    dt = float(x)
    return np.arange(length, dtype=float) * dt


def _normalize_window_length(length: int, window_length: int) -> int:
    window = max(3, int(window_length))
    if window % 2 == 0:
        window += 1
    if window >= length:
        window = length - 1 if length % 2 == 0 else length
    if window < 3:
        window = 3
    if window % 2 == 0:
        window += 1
    return min(window, length if length % 2 == 1 else length - 1)


def trajsegment_deriv(
    x: float | np.ndarray | list[float],
    speed: np.ndarray | list[float],
    window_length: int = 101,
    polyorder: int = 3,
    thres_dict: dict[str, float] | None = None,
    regime_mapper=map_segment_to_regime_index,
    min_duration: float = 5.0,
) -> np.ndarray:
    """
    Return (n_segments, 7):
    [start_idx, end_idx, start_time, end_time, regime_idx, slope, avg_speed]
    """
    speed_arr = np.asarray(speed, dtype=float)
    if speed_arr.ndim != 1:
        raise ValueError("speed must be a 1D array")
    if speed_arr.shape[0] < 5:
        return np.empty((0, 7), dtype=float)

    if thres_dict is None:
        thres_dict = {
            "acc": 0.1,
            "dec": -0.1,
            "acute_acc": 1.5,
            "acute_dec": -1.5,
            "mod_acc": 0.7,
            "mod_dec": -0.7,
        }

    time = _time_axis(x, speed_arr.shape[0])
    window = _normalize_window_length(speed_arr.shape[0], window_length)
    order = min(int(polyorder), max(1, window - 1))

    smooth = savgol_filter(speed_arr, window, order)
    deriv = np.gradient(smooth, time)
    raw_turns = np.where(np.diff(np.sign(deriv)) != 0)[0] + 1
    if raw_turns.size == 0:
        slope = float((smooth[-1] - smooth[0]) / max(1e-6, (time[-1] - time[0])))
        avg_speed = float(np.mean(smooth))
        regime = int(regime_mapper(slope, avg_speed, thres_dict))
        return np.array([[0, len(speed_arr) - 1, time[0], time[-1], regime, slope, avg_speed]], dtype=float)

    sig = np.array(
        [abs(deriv[idx + 1] - deriv[idx - 1]) if 1 < idx < (len(deriv) - 1) else 0.0 for idx in raw_turns]
    )

    filtered_turns: list[int] = []
    i = 0
    while i < len(raw_turns):
        current_idx = raw_turns[i]
        current_t = time[current_idx]
        j = i + 1
        while j < len(raw_turns) and (time[raw_turns[j]] - current_t) < min_duration:
            j += 1
        if j == i + 1:
            filtered_turns.append(int(current_idx))
            i += 1
            continue
        best = i + int(np.argmax(sig[i:j]))
        filtered_turns.append(int(raw_turns[best]))
        i = j

    filtered_turns.append(len(speed_arr) - 1)

    segments: list[list[float]] = []
    start_idx = 0
    for end_idx in filtered_turns:
        if end_idx - start_idx < 2:
            start_idx = end_idx
            continue

        t_seg = time[start_idx : end_idx + 1]
        v_seg = smooth[start_idx : end_idx + 1]
        slope = float((v_seg[-1] - v_seg[0]) / max(1e-6, (t_seg[-1] - t_seg[0])))
        avg_speed = float(np.mean(v_seg))
        regime = int(regime_mapper(slope, avg_speed, thres_dict))
        segments.append(
            [
                float(start_idx),
                float(end_idx),
                float(t_seg[0]),
                float(t_seg[-1]),
                float(regime),
                slope,
                avg_speed,
            ]
        )
        start_idx = end_idx

    return np.array(segments, dtype=float) if segments else np.empty((0, 7), dtype=float)


def split_starting_stopping_segments(
    movements: np.ndarray,
    dt: float,
    thres_dict: dict[str, float],
    window_length: int = 101,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    all_segments: list[np.ndarray] = []
    stopping: list[np.ndarray] = []
    starting: list[np.ndarray] = []

    for veh_idx in range(movements.shape[0]):
        seg = trajsegment_deriv(
            x=dt,
            speed=movements[veh_idx, :, 1],
            window_length=window_length,
            thres_dict=thres_dict,
            regime_mapper=map_segment_to_regime_index,
        )
        all_segments.append(seg)
        if seg.size == 0:
            stopping.append(np.empty((0, 7), dtype=float))
            starting.append(np.empty((0, 7), dtype=float))
            continue

        stopping.append(seg[np.isin(seg[:, 4], DEC_REGIME_IDX)])
        starting.append(seg[np.isin(seg[:, 4], ACC_REGIME_IDX)])

    return all_segments, stopping, starting


__all__ = ["trajsegment_deriv", "split_starting_stopping_segments"]
