from __future__ import annotations

from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np

from .model_tester import ModelEvalResult


def plot_metric_histograms(
    results: Mapping[str, ModelEvalResult],
    metrics: Sequence[str] = ("RMSE", "MAE", "MSE"),
    bins: int = 50,
    ranges: Mapping[str, tuple[float, float]] | None = None,
) -> plt.Figure:
    model_names = list(results.keys())
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:brown"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 4))
    if len(metrics) == 1:
        axes = [axes]

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        for model_idx, model_name in enumerate(model_names):
            values = results[model_name].metrics.get(metric, [])
            if not values:
                continue
            hist_range = None if ranges is None else ranges.get(metric)
            ax.hist(
                values,
                bins=bins,
                range=hist_range,
                alpha=0.3,
                color=colors[model_idx % len(colors)],
                label=model_name,
            )
        ax.set_title(metric)
        ax.set_xlabel(metric)
        ax.set_ylabel("Count")
        ax.grid(alpha=0.3)

    axes[-1].legend()
    fig.tight_layout()
    return fig


def plot_error_evolution(
    results: Mapping[str, ModelEvalResult],
    dt: float,
    boxplot_seconds: Sequence[int] = (10, 20, 30, 40, 50, 60, 70, 80),
) -> plt.Figure:
    fig, axes = plt.subplots(2, 1, figsize=(10, 7))
    ax_curve, ax_box = axes

    for model_name, result in results.items():
        if not result.errors:
            continue
        error = np.stack(result.errors, axis=0)
        mean = error.mean(axis=0)
        std = error.std(axis=0)
        time = np.arange(1, mean.shape[0] + 1) * dt
        ax_curve.plot(time, mean, label=model_name)
        ax_curve.fill_between(time, mean - std, mean + std, alpha=0.15)

    ax_curve.set_xlabel("Prediction Horizon (s)")
    ax_curve.set_ylabel("Absolute Position Error")
    ax_curve.set_title("Error Evolution")
    ax_curve.grid(alpha=0.3)
    ax_curve.legend(loc="upper left")

    offsets = np.linspace(-2.5, 2.5, num=max(1, len(results)))
    width = 1.1
    for model_idx, (model_name, result) in enumerate(results.items()):
        if not result.errors:
            continue
        error = np.stack(result.errors, axis=0)
        positions: list[float] = []
        values: list[np.ndarray] = []
        for sec in boxplot_seconds:
            index = int(round(sec / dt))
            if index >= error.shape[1]:
                continue
            positions.append(sec + offsets[model_idx])
            values.append(error[:, index])
        if not values:
            continue
        ax_box.boxplot(
            values,
            positions=positions,
            widths=width,
            patch_artist=True,
            boxprops={"alpha": 0.25},
            medianprops={"linewidth": 1.5},
            showfliers=False,
        )

    ax_box.set_xticks(list(boxplot_seconds))
    ax_box.set_xlabel("Prediction Horizon (s)")
    ax_box.set_ylabel("Absolute Position Error")
    ax_box.set_title("Error Distribution by Horizon")
    ax_box.grid(alpha=0.3)

    fig.tight_layout()
    return fig


def plot_trajectory_comparison(
    true_self: np.ndarray,
    true_leader: np.ndarray,
    predictions: Mapping[str, np.ndarray],
    dt: float,
) -> plt.Figure:
    labels = ["Position", "Velocity", "Acceleration", "Spacing"]
    fig, axes = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
    time = np.arange(true_self.shape[0]) * dt

    for axis_idx in range(3):
        axes[axis_idx].plot(time, true_self[:, axis_idx], label="true_follower", linewidth=2)
        for name, pred in predictions.items():
            axes[axis_idx].plot(time, pred[:, axis_idx], linestyle="--", label=name)
        axes[axis_idx].set_ylabel(labels[axis_idx])
        axes[axis_idx].grid(alpha=0.3)

    true_spacing = true_leader[:, 0] - true_self[:, 0]
    axes[3].plot(time, true_spacing, label="true_spacing", linewidth=2)
    for name, pred in predictions.items():
        spacing = true_leader[:, 0] - pred[:, 0]
        axes[3].plot(time, spacing, linestyle="--", label=f"{name}_spacing")
    axes[3].set_ylabel(labels[3])
    axes[3].set_xlabel("Time (s)")
    axes[3].grid(alpha=0.3)

    axes[0].legend(loc="upper left", ncol=2)
    fig.tight_layout()
    return fig
