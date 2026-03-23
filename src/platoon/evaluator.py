from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .analysis import split_starting_stopping_segments
from .metrics.fd import Parallelogram, compute_edie_qkv_parallelogram_matrix, query_proj_from_intersections
from .metrics.platoon_metrics import PlatoonMetrics
from .metrics.wave_metrics import AmpFactor, Wave

WAVE_COLUMNS = ["wave_id", "from", "to", "t_lead", "t_foll", "dx", "dt", "wave_speed"]


def filter_chain(chain: pd.DataFrame, wave_speed_floor: float, min_points: int) -> pd.DataFrame:
    if chain.empty:
        return chain
    out = chain.copy()
    out.loc[out["wave_speed"] < wave_speed_floor, "wave_speed"] = np.nan
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out[out.groupby("wave_id")["wave_speed"].transform("count") >= int(min_points)]
    return out.reset_index(drop=True)


def generate_wave_chain(
    movements: np.ndarray,
    segments: list[np.ndarray],
    dt: float,
    factor: float,
    tracking_tolerance: float,
) -> pd.DataFrame:
    if not segments:
        return pd.DataFrame(columns=WAVE_COLUMNS)

    segment_times: list[np.ndarray] = []
    for seg in segments:
        if seg.size == 0:
            segment_times.append(np.empty((0, 2), dtype=float))
            continue
        segment_times.append(np.asarray(seg[:, [2, 3]], dtype=float))

    waves = Wave.wave_speed(movements, segment_times, dt=dt, factor=factor)
    if len(waves) == 0:
        return pd.DataFrame(columns=WAVE_COLUMNS)

    chain = Wave.track_waves(
        waves,
        num_veh=movements.shape[0],
        time_tolerance=tracking_tolerance,
        columns=WAVE_COLUMNS,
    )
    if chain.size == 0:
        return pd.DataFrame(columns=WAVE_COLUMNS)
    return pd.DataFrame(chain, columns=WAVE_COLUMNS)


def calc_wave_velocities(movements: np.ndarray, chain: pd.DataFrame, dt: float, time_ahead: float) -> tuple[list[float], list[float]]:
    if chain.empty or "wave_id" not in chain.columns:
        return [], []
    wave_metrics = Wave()
    wave_speeds: list[float] = []
    veh_speeds: list[float] = []
    for wave_id in chain["wave_id"].unique():
        wave_rows = chain[chain["wave_id"] == wave_id]
        wv, vv = wave_metrics.wave_velocity(wave_rows, movements, dt=dt, time_ahead=time_ahead)
        wave_speeds.append(float(wv))
        veh_speeds.append(float(vv))
    return wave_speeds, veh_speeds


def calc_amp_factors(movements: np.ndarray, chain: pd.DataFrame, dt: float, time_shift: float) -> tuple[list[float], list[float], list[float]]:
    if chain.empty or "wave_id" not in chain.columns:
        return [], [], []
    amps: list[float] = []
    head_dv: list[float] = []
    tail_dv: list[float] = []
    for wave_id in chain["wave_id"].unique():
        wave_rows = chain[chain["wave_id"] == wave_id].to_numpy(dtype=float)
        af = AmpFactor.calc_amp_factor(movements, wave_rows, time_shift=time_shift, dt=dt)
        if af.size == 0:
            continue
        amps.append(float(af[0, -1]))
        head_dv.append(float(af[0, 5]))
        tail_dv.append(float(af[0, 6]))
    return amps, head_dv, tail_dv


@dataclass
class EvaluationResult:
    metrics: dict[str, Any]

    def summary(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for key, value in self.metrics.items():
            if np.isscalar(value):
                out[key] = float(value)
                continue
            arr = np.asarray(value, dtype=float)
            if arr.size == 0:
                out[key] = float("nan")
            else:
                out[key] = float(np.nanmean(arr))
        return out


def evaluate_simulation(
    movements: np.ndarray,
    evaluation_config: dict[str, Any],
    dt: float,
) -> EvaluationResult:
    thres_dict = {
        "acc": 0.1,
        "dec": -0.1,
        "acute_acc": 1.5,
        "acute_dec": -1.5,
        "mod_acc": 0.7,
        "mod_dec": -0.7,
    }

    _, stopping_segments, starting_segments = split_starting_stopping_segments(
        movements=movements,
        dt=dt,
        thres_dict=thres_dict,
        window_length=101,
    )

    wave_factor = float(evaluation_config.get("wave_lookahead_factor", 40.0))
    tracking_tol = float(evaluation_config.get("wave_tracking_tolerance", 5.0))
    speed_floor = float(evaluation_config.get("wave_speed_floor", -10.0))
    min_points = int(evaluation_config.get("wave_min_chain_points", 5))
    amp_shift = float(evaluation_config.get("amp_time_shift", 1.0))

    stopping_chain = filter_chain(
        generate_wave_chain(movements, stopping_segments, dt, wave_factor, tracking_tol),
        wave_speed_floor=speed_floor,
        min_points=min_points,
    )
    starting_chain = filter_chain(
        generate_wave_chain(movements, starting_segments, dt, wave_factor, tracking_tol),
        wave_speed_floor=speed_floor,
        min_points=min_points,
    )

    stop_wave, stop_veh = calc_wave_velocities(movements, stopping_chain, dt=dt, time_ahead=1.0)
    start_wave, start_veh = calc_wave_velocities(movements, starting_chain, dt=dt, time_ahead=1.0)

    stop_amp, stop_head_dv, stop_tail_dv = calc_amp_factors(movements, stopping_chain, dt=dt, time_shift=amp_shift)
    start_amp, start_head_dv, start_tail_dv = calc_amp_factors(movements, starting_chain, dt=dt, time_shift=amp_shift)

    offset = float(evaluation_config.get("offset", 700.0))
    speed_threshold = float(evaluation_config.get("speed_threshold", 1.0))

    metrics: dict[str, Any] = {}
    metrics["standstill_time"] = float(PlatoonMetrics.low_speed(movements, speed_threshold, time_step=dt, offset=offset))
    metrics["last_veh_stand_time"] = float(
        PlatoonMetrics.total_stopped_time_last_vehicle(movements, speed_threshold, time_step=dt, offset=offset)
    )
    metrics["delay"] = float(PlatoonMetrics.delay(movements, dt=dt, tau=1.5, offset=offset))
    metrics["stopping_wave_speed"] = stop_wave
    metrics["stopping_veh_speed"] = stop_veh
    metrics["starting_wave_speed"] = start_wave
    metrics["starting_veh_speed"] = start_veh
    metrics["stopping_amp_factor"] = stop_amp
    metrics["stop_head_delta_v"] = stop_head_dv
    metrics["stop_tail_delta_v"] = stop_tail_dv
    metrics["starting_amp_factor"] = start_amp
    metrics["start_head_delta_v"] = start_head_dv
    metrics["start_tail_delta_v"] = start_tail_dv
    metrics["vt_micro"] = float(PlatoonMetrics.vt_micro_fleet_L_per_km(movements, dt))

    wave_speed = float(evaluation_config.get("fd_wave_speed_kmh", -16.0)) / 3.6
    t_start = float(evaluation_config.get("fd_t_start", 25.0))
    t_end = float(evaluation_config.get("fd_t_end", 220.0))
    t_step = float(evaluation_config.get("fd_t_step", 8.0))
    t_length = float(evaluation_config.get("fd_t_length", 5.0))

    funds: list[list[float]] = []
    for t_center in np.arange(t_start, t_end, t_step):
        try:
            t_c, x_c, h_t, l_x, given_speed = query_proj_from_intersections(
                movements,
                t_center=t_center,
                t_length=t_length,
                dt=dt,
                wave_speed=wave_speed,
            )
            para = Parallelogram.from_proj_swapped(
                t_center=t_c,
                x_center=x_c,
                H_t=h_t,
                L_x=l_x,
                wave_speed=wave_speed,
                given_speed=given_speed,
            )
            q, k, v = compute_edie_qkv_parallelogram_matrix(
                car_trajs=movements[:, :, 0],
                dt_sample=dt,
                parallelogram=para,
            )
            funds.append([float(q), float(k), float(v)])
        except Exception:
            continue

    if funds:
        fund_np = np.asarray(funds, dtype=float)
        metrics["flow"] = fund_np[:, 0].tolist()
        metrics["density"] = fund_np[:, 1].tolist()
        metrics["speed"] = fund_np[:, 2].tolist()
    else:
        metrics["flow"] = []
        metrics["density"] = []
        metrics["speed"] = []

    return EvaluationResult(metrics=metrics)


__all__ = [
    "WAVE_COLUMNS",
    "filter_chain",
    "generate_wave_chain",
    "calc_wave_velocities",
    "calc_amp_factors",
    "EvaluationResult",
    "evaluate_simulation",
]
