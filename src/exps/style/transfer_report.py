"""
Style-transfer evaluation loops and distribution plots.
"""

from __future__ import annotations

from time import perf_counter
from typing import Any, Callable, Mapping, Protocol

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from tqdm.auto import tqdm

from ...schema import CFNAMES as CF
from ...utils.logger import logger


MetricSeries = dict[str, list[float]]
ClusterTransferResult = dict[tuple[int, int], MetricSeries]
FollowerTransferResult = dict[tuple[int, str], MetricSeries]


class StyleCaseLike(Protocol):
    metrics: Mapping[str, float]


class StyleCaseRunner(Protocol):
    def run_sample(
        self,
        data: Any,
        embedding: np.ndarray,
        *,
        sample: int,
        start_time: int = 60,
        lead_len: float | None = None,
        distance_offset: float = 0.0,
    ) -> StyleCaseLike:
        ...


def _empty_metric_series() -> MetricSeries:
    return {"thw": [], "react": []}


def _select_sample_indices(total: int, max_samples: int, rng: np.random.Generator) -> np.ndarray:
    if total <= 0:
        return np.zeros((0,), dtype=np.int64)
    if max_samples <= 0 or max_samples >= total:
        return np.arange(total, dtype=np.int64)
    return np.sort(rng.choice(total, size=max_samples, replace=False).astype(np.int64))


def _clone_datapack_rows(datapack: Any) -> Any:
    total = int(datapack.data.shape[0])
    return datapack.select_rows(np.arange(total, dtype=np.int64))


def run_cluster_transfer_grid(
    *,
    runner: StyleCaseRunner,
    data: Any,
    filter_by_cluster: Callable[..., Any],
    cluster_labels: np.ndarray,
    centroids: np.ndarray,
    start_time: int,
    is_truck: np.ndarray | None,
    keep_truck: bool,
    max_samples_per_cluster: int,
    random_seed: int,
    distance_offset: float = 0.0,
    show_progress: bool = False,
) -> ClusterTransferResult:
    t0 = perf_counter()
    centroids_arr = np.asarray(centroids, dtype=np.float32)
    if centroids_arr.ndim != 2:
        raise ValueError(f"centroids must be 2D, got shape={centroids_arr.shape}")

    k = int(centroids_arr.shape[0])
    logger.info(
        "Run cluster transfer grid start | "
        f"clusters={k} data_samples={int(data.data.shape[0])} max_samples_per_cluster={max_samples_per_cluster}"
    )
    rng = np.random.default_rng(int(random_seed))
    out: ClusterTransferResult = {}

    cluster_iter = range(k)
    if show_progress:
        cluster_iter = tqdm(cluster_iter, desc="Cluster transfer", leave=False)

    for cluster in cluster_iter:
        try:
            filtered = filter_by_cluster(
                data=data,
                cluster_labels=cluster_labels,
                target_cluster=cluster,
                is_truck=is_truck,
                keep_truck=keep_truck,
            )
        except ValueError:
            for compare_cluster in range(k):
                out[(cluster, compare_cluster)] = _empty_metric_series()
            continue

        total_samples = int(filtered.data.shape[0])
        logger.info(f"Cluster transfer filtered | cluster={cluster} samples={total_samples}")
        sample_indices = _select_sample_indices(total_samples, max_samples_per_cluster, rng)

        compare_iter = range(k)
        if show_progress:
            compare_iter = tqdm(compare_iter, desc=f"Cluster {cluster + 1}/{k}", leave=False)

        for compare_cluster in compare_iter:
            embedding = centroids_arr[compare_cluster][np.newaxis, :]
            metrics = _empty_metric_series()
            for sample in sample_indices.tolist():
                lead_len = float(filtered[sample, 0, CF.LEAD_L]) if CF.LEAD_L in filtered.names else 0.0
                result = runner.run_sample(
                    data=filtered,
                    embedding=embedding,
                    sample=int(sample),
                    start_time=int(start_time),
                    lead_len=lead_len,
                    distance_offset=distance_offset,
                )
                for metric_name in metrics:
                    value = result.metrics.get(metric_name)
                    if value is not None and np.isfinite(value):
                        metrics[metric_name].append(float(value))
            out[(cluster, compare_cluster)] = metrics

    logger.info(f"Run cluster transfer grid done | pairs={len(out)} elapsed={perf_counter() - t0:.2f}s")
    return out


def run_follower_type_substitution(
    *,
    runner: StyleCaseRunner,
    data: Any,
    filter_by_cluster: Callable[..., Any],
    cluster_labels: np.ndarray,
    centroids: np.ndarray,
    start_time: int,
    is_truck: np.ndarray | None,
    base_keep_truck: bool,
    truck_length_override: float,
    passenger_length_override: float,
    max_samples_per_cluster: int,
    random_seed: int,
    distance_offset: float = 0.0,
) -> FollowerTransferResult:
    centroids_arr = np.asarray(centroids, dtype=np.float32)
    if centroids_arr.ndim != 2:
        raise ValueError(f"centroids must be 2D, got shape={centroids_arr.shape}")

    k = int(centroids_arr.shape[0])
    rng = np.random.default_rng(int(random_seed) + 101)
    out: FollowerTransferResult = {}

    follower_options = (
        ("truck", float(truck_length_override)),
        ("notruck", float(passenger_length_override)),
    )

    for cluster in range(k):
        try:
            filtered = filter_by_cluster(
                data=data,
                cluster_labels=cluster_labels,
                target_cluster=cluster,
                is_truck=is_truck,
                keep_truck=base_keep_truck,
            )
        except ValueError:
            for follower_type, _ in follower_options:
                out[(cluster, follower_type)] = _empty_metric_series()
            continue

        total_samples = int(filtered.data.shape[0])
        sample_indices = _select_sample_indices(total_samples, max_samples_per_cluster, rng)
        embedding = centroids_arr[cluster][np.newaxis, :]

        for follower_type, length_override in follower_options:
            eval_data = _clone_datapack_rows(filtered)
            if CF.SELF_L in eval_data.names:
                eval_data.replace_col(
                    np.ones_like(eval_data.data[:, :, 0], dtype=np.float32) * float(length_override),
                    CF.SELF_L,
                )
            metrics = _empty_metric_series()
            for sample in sample_indices.tolist():
                lead_len = float(eval_data[sample, 0, CF.LEAD_L]) if CF.LEAD_L in eval_data.names else 0.0
                result = runner.run_sample(
                    data=eval_data,
                    embedding=embedding,
                    sample=int(sample),
                    start_time=int(start_time),
                    lead_len=lead_len,
                    distance_offset=distance_offset,
                )
                for metric_name in metrics:
                    value = result.metrics.get(metric_name)
                    if value is not None and np.isfinite(value):
                        metrics[metric_name].append(float(value))
            out[(cluster, follower_type)] = metrics

    return out


def _kde_density(values: np.ndarray, xs: np.ndarray) -> np.ndarray | None:
    if values.size < 2:
        return None
    if np.allclose(values, values[0]):
        return None
    try:
        kde = gaussian_kde(values)
        return kde(xs)
    except Exception:
        return None


def _filtered_metric_values(
    source: Mapping[str, list[float]],
    metric_name: str,
    xlim: tuple[float, float],
) -> np.ndarray:
    values = np.asarray(source.get(metric_name, []), dtype=np.float32)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return values
    xmin, xmax = xlim
    return values[(values >= xmin) & (values <= xmax)]


def plot_cluster_vs_embedding_styles(
    *,
    results: ClusterTransferResult,
    num_clusters: int,
    xlim: tuple[float, float],
    show_progress: bool = False,
) -> plt.Figure:
    t0 = perf_counter()
    xmin, xmax = xlim
    xs = np.linspace(xmin, xmax, 300, dtype=np.float32)
    metric_defs = (("thw", r"$\bar h$"), ("react", r"$\bar \tau$"))
    colors = plt.get_cmap("tab10").colors

    max_density = 0.0
    for metric_name, _ in metric_defs:
        density_cluster_iter = range(num_clusters)
        if show_progress:
            density_cluster_iter = tqdm(density_cluster_iter, desc=f"KDE density {metric_name}", leave=False)
        for cluster in density_cluster_iter:
            for compare_cluster in range(num_clusters):
                values = _filtered_metric_values(results.get((cluster, compare_cluster), {}), metric_name, xlim)
                density = _kde_density(values, xs)
                if density is not None:
                    max_density = max(max_density, float(np.max(density)))

    fig, axes = plt.subplots(2, num_clusters, figsize=(4.5 * num_clusters, 6), sharey=True)
    if num_clusters == 1:
        axes = np.asarray(axes).reshape(2, 1)

    for row_idx, (metric_name, title_metric) in enumerate(metric_defs):
        plot_cluster_iter = range(num_clusters)
        if show_progress:
            plot_cluster_iter = tqdm(plot_cluster_iter, desc=f"Plot report {metric_name}", leave=False)
        for cluster in plot_cluster_iter:
            ax = axes[row_idx, cluster]
            for compare_cluster in range(num_clusters):
                values = _filtered_metric_values(results.get((cluster, compare_cluster), {}), metric_name, xlim)
                density = _kde_density(values, xs)
                if density is None:
                    continue
                color = colors[compare_cluster % len(colors)]
                ax.plot(xs, density, color=color, linewidth=2, label=f"P{compare_cluster + 1}")
                ax.axvline(float(np.mean(values)), color=color, linestyle="--", alpha=0.5)

            ax.set_xlim(xmin, xmax)
            if max_density > 0:
                ax.set_ylim(0.0, max_density * 1.05)
            ax.set_title(f"{title_metric} | True Cluster P{cluster + 1}")
            if metric_name == "thw":
                ax.set_xlabel("THW (s)")
            else:
                ax.set_xlabel("Reaction Time (s)")
            if cluster == 0:
                ax.set_ylabel("Density")
            if row_idx == 0 and cluster == num_clusters - 1:
                ax.legend(title="Substituted Style", loc="upper right", fontsize=9)
            ax.grid(True, alpha=0.25)

    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    logger.info(
        "Plot cluster-vs-embedding styles done | "
        f"clusters={num_clusters} xlim={xlim} elapsed={perf_counter() - t0:.2f}s"
    )
    return fig


def plot_follower_type_styles(
    *,
    results: FollowerTransferResult,
    num_clusters: int,
    xlim: tuple[float, float],
    show_progress: bool = False,
) -> plt.Figure:
    t0 = perf_counter()
    xmin, xmax = xlim
    xs = np.linspace(xmin, xmax, 300, dtype=np.float32)
    metric_defs = (("thw", r"$\bar h$"), ("react", r"$\bar \tau$"))
    followers = (("truck", "Truck"), ("notruck", "Passenger Car"))
    colors = ("tab:blue", "tab:orange")

    max_density = 0.0
    for metric_name, _ in metric_defs:
        density_cluster_iter = range(num_clusters)
        if show_progress:
            density_cluster_iter = tqdm(density_cluster_iter, desc=f"Follower KDE {metric_name}", leave=False)
        for cluster in density_cluster_iter:
            for follower_key, _ in followers:
                values = _filtered_metric_values(results.get((cluster, follower_key), {}), metric_name, xlim)
                density = _kde_density(values, xs)
                if density is not None:
                    max_density = max(max_density, float(np.max(density)))

    fig, axes = plt.subplots(2, num_clusters, figsize=(4.5 * num_clusters, 6), sharey=True)
    if num_clusters == 1:
        axes = np.asarray(axes).reshape(2, 1)

    for row_idx, (metric_name, title_metric) in enumerate(metric_defs):
        plot_cluster_iter = range(num_clusters)
        if show_progress:
            plot_cluster_iter = tqdm(plot_cluster_iter, desc=f"Plot follower {metric_name}", leave=False)
        for cluster in plot_cluster_iter:
            ax = axes[row_idx, cluster]
            for idx, (follower_key, follower_label) in enumerate(followers):
                values = _filtered_metric_values(results.get((cluster, follower_key), {}), metric_name, xlim)
                density = _kde_density(values, xs)
                if density is None:
                    continue
                ax.plot(xs, density, color=colors[idx], linewidth=2, label=follower_label)
                ax.axvline(float(np.mean(values)), color=colors[idx], linestyle="--", alpha=0.5)

            ax.set_xlim(xmin, xmax)
            if max_density > 0:
                ax.set_ylim(0.0, max_density * 1.05)
            ax.set_title(f"{title_metric} | Cluster P{cluster + 1}")
            if metric_name == "thw":
                ax.set_xlabel("THW (s)")
            else:
                ax.set_xlabel("Reaction Time (s)")
            if cluster == 0:
                ax.set_ylabel("Density")
            if row_idx == 0 and cluster == num_clusters - 1:
                ax.legend(title="Substituted Follower", loc="upper right", fontsize=9)
            ax.grid(True, alpha=0.25)

    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    logger.info(
        "Plot follower-type styles done | "
        f"clusters={num_clusters} xlim={xlim} elapsed={perf_counter() - t0:.2f}s"
    )
    return fig


def summarize_cluster_transfer(results: ClusterTransferResult) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for (cluster, compare_cluster), metrics in sorted(results.items()):
        thw = np.asarray(metrics.get("thw", []), dtype=np.float32)
        react = np.asarray(metrics.get("react", []), dtype=np.float32)
        rows.append(
            {
                "cluster": int(cluster),
                "compare_cluster": int(compare_cluster),
                "count": int(max(thw.size, react.size)),
                "thw_mean": float(np.mean(thw)) if thw.size > 0 else float("nan"),
                "thw_std": float(np.std(thw)) if thw.size > 0 else float("nan"),
                "react_mean": float(np.mean(react)) if react.size > 0 else float("nan"),
                "react_std": float(np.std(react)) if react.size > 0 else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def summarize_follower_transfer(results: FollowerTransferResult) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for (cluster, follower_type), metrics in sorted(results.items()):
        thw = np.asarray(metrics.get("thw", []), dtype=np.float32)
        react = np.asarray(metrics.get("react", []), dtype=np.float32)
        rows.append(
            {
                "cluster": int(cluster),
                "follower_type": str(follower_type),
                "count": int(max(thw.size, react.size)),
                "thw_mean": float(np.mean(thw)) if thw.size > 0 else float("nan"),
                "thw_std": float(np.std(thw)) if thw.size > 0 else float("nan"),
                "react_mean": float(np.mean(react)) if react.size > 0 else float("nan"),
                "react_std": float(np.std(react)) if react.size > 0 else float("nan"),
            }
        )
    return pd.DataFrame(rows)


__all__ = [
    "ClusterTransferResult",
    "FollowerTransferResult",
    "run_cluster_transfer_grid",
    "run_follower_type_substitution",
    "plot_cluster_vs_embedding_styles",
    "plot_follower_type_styles",
    "summarize_cluster_transfer",
    "summarize_follower_transfer",
]
