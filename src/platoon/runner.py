from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from .builder import PlatoonSimulationBuilder
from .config_loader import get_platoon_configs
from .evaluator import EvaluationResult, evaluate_simulation
from .statistics import run_anova


def append_results(all_results: dict[str, list[float]], metrics: dict[str, Any]) -> dict[str, list[float]]:
    for key, value in metrics.items():
        all_results.setdefault(key, [])
        if np.isscalar(value):
            all_results[key].append(float(value))
            continue
        arr = np.asarray(value, dtype=float).reshape(-1)
        if arr.size == 0:
            continue
        all_results[key].extend(arr.tolist())
    return all_results


def _mean_or_nan(values: list[float]) -> float:
    if not values:
        return float("nan")
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float("nan")
    return float(np.nanmean(arr))


def summarize_group_metrics(all_group_results: dict[str, dict[str, list[float]]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group_name, metrics in all_group_results.items():
        row: dict[str, Any] = {"group": group_name}
        for key, values in metrics.items():
            row[key] = _mean_or_nan(values)
        rows.append(row)
    return pd.DataFrame(rows)


def _build_fixed_composition(
    builder: PlatoonSimulationBuilder,
    group_cfg: dict[str, Any],
) -> list[str]:
    composition = group_cfg.get("composition")
    if composition is None:
        raise ValueError("Fixed group requires 'composition'.")
    if isinstance(composition, list):
        return [str(x) for x in composition]
    if isinstance(composition, dict):
        return builder.build_composition_from_counts({str(k): int(v) for k, v in composition.items()})
    raise TypeError(f"Unsupported fixed composition type: {type(composition).__name__}")


def _veh_lens_for_composition(composition: list[str], group_cfg: dict[str, Any]) -> list[float]:
    if "veh_lens" in group_cfg:
        raw = list(group_cfg["veh_lens"])
        if len(raw) != len(composition):
            raise ValueError("veh_lens length must match composition length.")
        return [float(v) for v in raw]

    car_len = float(group_cfg.get("car_length", 5.0))
    truck_len = float(group_cfg.get("truck_length", 9.0))
    return [truck_len if label.upper().startswith("T") else car_len for label in composition]


@dataclass
class RunOutput:
    raw_results: dict[str, dict[str, list[float]]]
    summary: pd.DataFrame
    anova: dict[str, dict[str, Any]]
    output_dir: Path


def run_platoon_experiments(configs: dict[str, Any] | None = None) -> RunOutput:
    cfg = get_platoon_configs() if configs is None else configs
    simulation_config = dict(cfg["simulation_config"])
    newell_config = dict(cfg["newell_config"])
    evaluation_config = dict(cfg["evaluation_config"])
    experiments_cfg = dict(cfg["experiments"])
    groups_cfg = dict(experiments_cfg.get("groups", {}))
    enabled_groups = list(experiments_cfg.get("enabled_groups", groups_cfg.keys()))

    repetitions = int(simulation_config.get("repetitions", 5))
    leader_ids = [int(x) for x in simulation_config.get("leader_ids", [0])]
    dt = float(simulation_config.get("dt", 0.1))

    builder = PlatoonSimulationBuilder(simulation_config, newell_config)
    all_results: dict[str, dict[str, list[float]]] = {}

    for group_name in enabled_groups:
        if group_name not in groups_cfg:
            continue
        group_cfg = dict(groups_cfg[group_name])
        mode = str(group_cfg.get("mode", "fixed")).lower()

        group_results: dict[str, list[float]] = {}
        total_iter = len(leader_ids) * repetitions
        bar = tqdm(total=total_iter, desc=f"Platoon {group_name}")

        for leader_id in leader_ids:
            for _ in range(repetitions):
                if mode == "fixed":
                    composition = _build_fixed_composition(builder, group_cfg)
                elif mode == "mix":
                    distribution = dict(group_cfg.get("distribution", {}))
                    platoon_len = int(group_cfg.get("platoon_len", 16))
                    composition = builder.sample_mix_composition(distribution, platoon_len)
                else:
                    raise ValueError(f"Unsupported experiment mode: {mode}")

                veh_lens = _veh_lens_for_composition(composition, group_cfg)
                movements = builder.simulate(composition=composition, leader_id=leader_id, veh_lens=veh_lens)
                eva: EvaluationResult = evaluate_simulation(movements, evaluation_config=evaluation_config, dt=dt)
                append_results(group_results, eva.metrics)
                bar.update(1)

        bar.close()
        all_results[group_name] = group_results

    summary = summarize_group_metrics(all_results)
    anova = run_anova(all_results)
    out_dir = Path(str(simulation_config.get("output_dir", "models/platoon_results")))
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = out_dir / "summary.csv"
    summary.to_csv(summary_path, index=False)

    for group_name, group_metrics in all_results.items():
        (out_dir / f"{group_name}_metrics.json").write_text(
            json.dumps(group_metrics, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    (out_dir / "anova.json").write_text(json.dumps(anova, ensure_ascii=False, indent=2), encoding="utf-8")

    return RunOutput(
        raw_results=all_results,
        summary=summary,
        anova=anova,
        output_dir=out_dir,
    )


__all__ = ["RunOutput", "append_results", "summarize_group_metrics", "run_platoon_experiments"]
