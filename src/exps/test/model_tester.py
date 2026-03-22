from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from ..agent import Agent


MetricMap = dict[str, list[float]]


@dataclass
class ModelEvalResult:
    model_name: str
    metrics: MetricMap
    errors: list[np.ndarray]
    start_time: int

    def summary(self) -> dict[str, float]:
        return {
            metric: float(np.mean(values)) if values else float("nan")
            for metric, values in self.metrics.items()
        }

    def as_dataframe(self, digits: int = 6) -> pd.DataFrame:
        rounded = {
            metric: [round(float(v), digits) for v in values]
            for metric, values in self.metrics.items()
        }
        return pd.DataFrame(rounded)


def _to_numpy(data: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(data, np.ndarray):
        return data
    return data.detach().cpu().numpy()


def _compute_metrics(
    pred: np.ndarray,
    true: np.ndarray,
    epsilon: float = 1e-8,
) -> dict[str, float]:
    diff = pred - true
    mse = float(np.mean(diff ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(diff)))
    mape = float(np.mean(np.abs(diff / (true + epsilon))) * 100.0)
    smape = float(
        np.mean(2.0 * np.abs(diff) / (np.abs(pred) + np.abs(true) + epsilon)) * 100.0
    )
    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "MAPE": mape,
        "SMAPE": smape,
    }


def evaluate_rollout(
    model_name: str,
    agent: Agent,
    num_samples: int,
    build_x: Callable[[int], object],
    build_traj: Callable[[int], tuple[torch.Tensor, torch.Tensor]],
    pred_func: Callable,
    mask: Callable,
    start_time: int = 0,
    desc: str | None = None,
) -> ModelEvalResult:
    metrics: MetricMap = {
        "MSE": [],
        "RMSE": [],
        "MAE": [],
        "MAPE": [],
        "SMAPE": [],
    }
    errors: list[np.ndarray] = []

    iterator = range(num_samples)
    progress_desc = desc or f"Evaluating {model_name}"
    for idx in tqdm(iterator, desc=progress_desc):
        x_full = build_x(idx)
        self_traj, leader_traj = build_traj(idx)

        pred_self = agent.predict(
            x_full,
            self_traj,
            leader_traj,
            pred_func=pred_func,
            mask=mask,
        )

        pred_np = _to_numpy(pred_self)
        true_np = _to_numpy(self_traj)

        # Follow notebook behavior: evaluate follower position after warmup.
        pred_pos = pred_np[start_time:, 0]
        true_pos = true_np[start_time:, 0]

        if pred_pos.shape[0] != true_pos.shape[0]:
            raise ValueError(
                "Prediction/ground-truth length mismatch after warmup: "
                f"pred={pred_pos.shape[0]}, true={true_pos.shape[0]}. "
                "Adjust model start_time/start_step to align rollout windows."
            )
        point_metrics = _compute_metrics(pred_pos, true_pos)
        for key, value in point_metrics.items():
            metrics[key].append(value)

        errors.append(np.abs(true_pos - pred_pos))

    return ModelEvalResult(
        model_name=model_name,
        metrics=metrics,
        errors=errors,
        start_time=start_time,
    )


def save_result_csv(
    result: ModelEvalResult,
    output_dir: str | Path,
    save_errors: bool = True,
) -> tuple[Path, Path | None]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metric_path = out_dir / f"{result.model_name}_results.csv"
    result.as_dataframe().to_csv(metric_path, index=False)

    error_path: Path | None = None
    if save_errors and result.errors:
        error_path = out_dir / f"{result.model_name}_errors.npy"
        np.save(error_path, np.stack(result.errors, axis=0))

    return metric_path, error_path


def summarize_results(results: Mapping[str, ModelEvalResult]) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for model_name, result in results.items():
        row: dict[str, float | str] = {"model": model_name}
        row.update(result.summary())
        rows.append(row)
    return pd.DataFrame(rows)


def save_result_bundle(
    results: Mapping[str, ModelEvalResult],
    output_dir: str | Path,
    save_errors: bool = True,
    merge_with_existing_summary: bool = False,
) -> tuple[Path, dict[str, tuple[Path, Path | None]]]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = summarize_results(results)
    summary_path = out_dir / "summary.csv"

    if merge_with_existing_summary and summary_path.exists():
        try:
            old_summary = pd.read_csv(summary_path)
        except Exception:
            old_summary = pd.DataFrame()

        if not old_summary.empty and "model" in old_summary.columns and "model" in summary.columns:
            old_summary["model"] = old_summary["model"].astype(str)
            summary["model"] = summary["model"].astype(str)
            keep_old = old_summary.loc[~old_summary["model"].isin(summary["model"])]
            summary = pd.concat([keep_old, summary], ignore_index=True)

    summary.to_csv(summary_path, index=False)

    per_model_paths: dict[str, tuple[Path, Path | None]] = {}
    for model_name, result in results.items():
        per_model_paths[model_name] = save_result_csv(
            result=result,
            output_dir=out_dir,
            save_errors=save_errors,
        )

    return summary_path, per_model_paths
