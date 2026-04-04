"""
Embedding clustering and visualization helpers for style pipeline reports.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture


@dataclass(frozen=True)
class ClusterResult:
    labels: np.ndarray
    centroids: np.ndarray


class EmbeddingClusterProtocol(Protocol):
    def __call__(
        self,
        *,
        tokens: np.ndarray,
        n_clusters: int,
        random_seed: int,
        options: Mapping[str, Any],
    ) -> ClusterResult: ...


@dataclass(frozen=True)
class ClusterGroup:
    name: str
    cluster_method: str
    sample_indices: np.ndarray
    labels: np.ndarray
    centroids: np.ndarray
    reduced: np.ndarray
    centroids_reduced: np.ndarray
    silhouette: float

    @property
    def num_clusters(self) -> int:
        return int(self.centroids.shape[0]) if self.centroids.ndim == 2 else 0


@dataclass(frozen=True)
class EmbeddingClusterReport:
    tokens: np.ndarray
    is_truck_leader: np.ndarray
    lead_length_threshold: float
    truck: ClusterGroup
    notruck: ClusterGroup


def _normalize_labels_and_centroids(
    *,
    tokens: np.ndarray,
    raw_labels: np.ndarray,
) -> ClusterResult:
    labels_raw = np.asarray(raw_labels, dtype=np.int64).reshape(-1)
    if labels_raw.shape[0] != tokens.shape[0]:
        raise ValueError(f"label size mismatch: labels={labels_raw.shape[0]} samples={tokens.shape[0]}")

    unique_labels = np.unique(labels_raw)
    label_map = {int(label): idx for idx, label in enumerate(unique_labels.tolist())}
    labels = np.array([label_map[int(label)] for label in labels_raw], dtype=np.int64)

    centroids = np.vstack(
        [np.mean(tokens[labels_raw == label], axis=0) for label in unique_labels]
    ).astype(np.float32, copy=False)
    return ClusterResult(labels=labels, centroids=centroids)


def _silhouette_or_nan(tokens: np.ndarray, labels: np.ndarray) -> float:
    if tokens.shape[0] < 2:
        return float("nan")
    num_labels = int(np.unique(labels).size)
    if num_labels < 2 or num_labels >= int(tokens.shape[0]):
        return float("nan")
    return float(silhouette_score(tokens, labels))


def _pca_project_2d(
    *,
    tokens: np.ndarray,
    centroids: np.ndarray,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if tokens.shape[0] == 0:
        empty = np.zeros((0, 2), dtype=np.float32)
        return empty, empty.copy()
    if tokens.shape[1] == 0:
        raise ValueError("tokens feature dimension must be > 0")

    n_components = min(2, int(tokens.shape[0]), int(tokens.shape[1]))
    pca = PCA(n_components=n_components, random_state=int(random_seed))

    reduced_raw = pca.fit_transform(tokens).astype(np.float32, copy=False)
    if centroids.shape[0] > 0:
        centroids_raw = pca.transform(centroids).astype(np.float32, copy=False)
    else:
        centroids_raw = np.zeros((0, n_components), dtype=np.float32)

    if n_components == 2:
        return reduced_raw, centroids_raw

    reduced = np.zeros((int(tokens.shape[0]), 2), dtype=np.float32)
    reduced[:, 0] = reduced_raw[:, 0]

    centroids_reduced = np.zeros((int(centroids.shape[0]), 2), dtype=np.float32)
    if centroids_raw.shape[0] > 0:
        centroids_reduced[:, 0] = centroids_raw[:, 0]
    return reduced, centroids_reduced


def cluster_embeddings_kmeans(
    *,
    tokens: np.ndarray,
    n_clusters: int,
    random_seed: int,
    options: Mapping[str, Any],
) -> ClusterResult:
    k = max(1, min(int(n_clusters), int(tokens.shape[0])))

    n_init_option = options.get("n_init", "auto")
    if isinstance(n_init_option, str) and n_init_option.strip().lower() == "auto":
        n_init: int | str = "auto"
    else:
        n_init = int(n_init_option)

    model = KMeans(
        n_clusters=k,
        random_state=int(random_seed),
        n_init=n_init,
        max_iter=int(options.get("max_iter", 300)),
        algorithm=str(options.get("algorithm", "lloyd")),
    )
    labels = model.fit_predict(tokens).astype(np.int64, copy=False)
    centroids = model.cluster_centers_.astype(np.float32, copy=False)
    return ClusterResult(labels=labels, centroids=centroids)


def cluster_embeddings_gmm(
    *,
    tokens: np.ndarray,
    n_clusters: int,
    random_seed: int,
    options: Mapping[str, Any],
) -> ClusterResult:
    k = max(1, min(int(n_clusters), int(tokens.shape[0])))
    covariance_type = str(options.get("covariance_type", "full")).strip().lower()
    if covariance_type not in {"full", "tied", "diag", "spherical"}:
        raise ValueError(
            "Unsupported GMM covariance_type="
            f"'{covariance_type}'. Expected one of: full, tied, diag, spherical."
        )

    model = GaussianMixture(
        n_components=k,
        covariance_type=covariance_type,
        reg_covar=float(options.get("reg_covar", 1e-6)),
        max_iter=int(options.get("max_iter", 300)),
        n_init=int(options.get("n_init", 1)),
        random_state=int(random_seed),
    )
    labels = model.fit_predict(tokens).astype(np.int64, copy=False)
    centroids = model.means_.astype(np.float32, copy=False)
    return ClusterResult(labels=labels, centroids=centroids)


def cluster_embeddings_hdbscan(
    *,
    tokens: np.ndarray,
    n_clusters: int,
    random_seed: int,
    options: Mapping[str, Any],
) -> ClusterResult:
    _ = n_clusters
    _ = random_seed

    try:
        import hdbscan  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "cluster.method='hdbscan' requires the optional package 'hdbscan'. "
            "Install it with: pip install hdbscan"
        ) from exc

    default_min_cluster_size = max(8, int(tokens.shape[0]) // 20) if tokens.shape[0] > 0 else 8

    min_samples_raw = options.get("min_samples")
    if min_samples_raw is None or (
        isinstance(min_samples_raw, str) and min_samples_raw.strip().lower() == "auto"
    ):
        min_samples = None
    else:
        min_samples = int(min_samples_raw)

    cluster_selection_method = str(options.get("cluster_selection_method", "eom")).strip().lower()
    if cluster_selection_method not in {"eom", "leaf"}:
        raise ValueError(
            "Unsupported HDBSCAN cluster_selection_method="
            f"'{cluster_selection_method}'. Expected one of: eom, leaf."
        )

    model = hdbscan.HDBSCAN(
        min_cluster_size=int(options.get("min_cluster_size", default_min_cluster_size)),
        min_samples=min_samples,
        metric=str(options.get("metric", "euclidean")).strip(),
        cluster_selection_method=cluster_selection_method,
        cluster_selection_epsilon=float(options.get("cluster_selection_epsilon", 0.0)),
        allow_single_cluster=bool(options.get("allow_single_cluster", False)),
    )
    raw_labels = model.fit_predict(tokens)
    return _normalize_labels_and_centroids(tokens=tokens, raw_labels=raw_labels)


_CLUSTERERS: dict[str, EmbeddingClusterProtocol] = {
    "kmeans": cluster_embeddings_kmeans,
    "gmm": cluster_embeddings_gmm,
    "hdbscan": cluster_embeddings_hdbscan,
}


def _resolve_cluster_protocol(
    cluster_protocol: Mapping[str, Any] | None,
) -> tuple[str, dict[str, Any]]:
    protocol = dict(cluster_protocol or {})
    method = str(protocol.get("method", "kmeans")).strip().lower()
    if method not in _CLUSTERERS:
        supported = ", ".join(sorted(_CLUSTERERS.keys()))
        raise ValueError(f"Unsupported cluster method '{method}'. Supported methods: {supported}.")

    method_payload = protocol.get(method, {})
    if method_payload is None:
        method_options: dict[str, Any] = {}
    elif isinstance(method_payload, Mapping):
        method_options = dict(method_payload)
    else:
        raise ValueError(
            f"cluster.{method} must be a table/dict, got type={type(method_payload).__name__}"
        )

    common_options = {
        key: value
        for key, value in protocol.items()
        if key not in {"method", "kmeans", "gmm", "hdbscan"}
    }
    return method, {**common_options, **method_options}


def _empty_group(name: str, feature_dim: int, *, cluster_method: str) -> ClusterGroup:
    empty_idx = np.zeros((0,), dtype=np.int64)
    empty_centroids = np.zeros((0, feature_dim), dtype=np.float32)
    empty_2d = np.zeros((0, 2), dtype=np.float32)
    return ClusterGroup(
        name=name,
        cluster_method=cluster_method,
        sample_indices=empty_idx,
        labels=empty_idx.copy(),
        centroids=empty_centroids,
        reduced=empty_2d,
        centroids_reduced=empty_2d.copy(),
        silhouette=float("nan"),
    )


def _cluster_group(
    *,
    name: str,
    tokens: np.ndarray,
    sample_indices: np.ndarray,
    n_clusters: int,
    random_seed: int,
    cluster_method: str,
    cluster_options: Mapping[str, Any],
) -> ClusterGroup:
    feature_dim = int(tokens.shape[1]) if tokens.ndim == 2 else 0
    if tokens.shape[0] == 0:
        return _empty_group(name, feature_dim=feature_dim, cluster_method=cluster_method)

    cluster_fn = _CLUSTERERS[cluster_method]
    result = cluster_fn(
        tokens=tokens,
        n_clusters=max(1, min(int(n_clusters), int(tokens.shape[0]))),
        random_seed=int(random_seed),
        options=cluster_options,
    )

    labels = np.asarray(result.labels, dtype=np.int64).reshape(-1)
    if labels.shape[0] != tokens.shape[0]:
        raise ValueError(
            f"{cluster_method} label size mismatch: labels={labels.shape[0]} samples={tokens.shape[0]}"
        )

    centroids = np.asarray(result.centroids, dtype=np.float32)
    if centroids.ndim != 2 or centroids.shape[1] != feature_dim:
        raise ValueError(
            f"{cluster_method} centroids shape mismatch: expected (*, {feature_dim}), got {centroids.shape}"
        )

    expected_num_clusters = int(np.unique(labels).size)
    if centroids.shape[0] != expected_num_clusters:
        raise ValueError(
            f"{cluster_method} centroid count mismatch: centroids={centroids.shape[0]} labels={expected_num_clusters}"
        )

    reduced, centroids_reduced = _pca_project_2d(
        tokens=tokens,
        centroids=centroids,
        random_seed=int(random_seed),
    )

    return ClusterGroup(
        name=name,
        cluster_method=cluster_method,
        sample_indices=sample_indices.astype(np.int64, copy=False),
        labels=labels,
        centroids=centroids,
        reduced=reduced,
        centroids_reduced=centroids_reduced,
        silhouette=_silhouette_or_nan(tokens, labels),
    )


def cluster_style_embeddings(
    *,
    tokens: np.ndarray,
    is_truck_leader: np.ndarray,
    lead_length_threshold: float,
    truck_clusters: int,
    notruck_clusters: int,
    random_seed: int,
    cluster_protocol: Mapping[str, Any] | None = None,
) -> EmbeddingClusterReport:
    tokens_arr = np.asarray(tokens, dtype=np.float32)
    if tokens_arr.ndim != 2:
        raise ValueError(f"tokens must be 2D, got shape={tokens_arr.shape}")

    truck_mask = np.asarray(is_truck_leader, dtype=bool).reshape(-1)
    if truck_mask.shape[0] != tokens_arr.shape[0]:
        raise ValueError(
            "is_truck_leader length mismatch: "
            f"{truck_mask.shape[0]} vs {tokens_arr.shape[0]}"
        )

    cluster_method, cluster_options = _resolve_cluster_protocol(cluster_protocol)
    notruck_mask = ~truck_mask

    truck_group = _cluster_group(
        name="truck",
        tokens=tokens_arr[truck_mask],
        sample_indices=np.flatnonzero(truck_mask),
        n_clusters=int(truck_clusters),
        random_seed=int(random_seed),
        cluster_method=cluster_method,
        cluster_options=cluster_options,
    )
    notruck_group = _cluster_group(
        name="notruck",
        tokens=tokens_arr[notruck_mask],
        sample_indices=np.flatnonzero(notruck_mask),
        n_clusters=int(notruck_clusters),
        random_seed=int(random_seed) + 17,
        cluster_method=cluster_method,
        cluster_options=cluster_options,
    )

    return EmbeddingClusterReport(
        tokens=tokens_arr,
        is_truck_leader=truck_mask,
        lead_length_threshold=float(lead_length_threshold),
        truck=truck_group,
        notruck=notruck_group,
    )


def plot_embedding_clusters(report: EmbeddingClusterReport) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=False, sharey=False)
    groups = (
        ("Truck-leading Clusters", report.truck, "tab10"),
        ("Passenger Car-leading Clusters", report.notruck, "tab10"),
    )

    for ax, (title, group, cmap_name) in zip(axes, groups):
        ax.set_title(title)
        ax.set_xlabel("PCA Component 1")
        ax.set_ylabel("PCA Component 2")
        ax.grid(True, alpha=0.3)

        if group.sample_indices.size == 0:
            ax.text(0.5, 0.5, "No samples", ha="center", va="center", transform=ax.transAxes)
            continue

        cmap = plt.get_cmap(cmap_name)
        for cluster_id in range(group.num_clusters):
            mask = group.labels == cluster_id
            if not np.any(mask):
                continue
            ax.scatter(
                group.reduced[mask, 0],
                group.reduced[mask, 1],
                s=24,
                alpha=0.75,
                color=cmap(cluster_id % 10),
                label=f"C{cluster_id}",
            )

        if group.centroids_reduced.size > 0:
            ax.scatter(
                group.centroids_reduced[:, 0],
                group.centroids_reduced[:, 1],
                c="black",
                marker="x",
                s=110,
                linewidths=2,
                label="Centroids",
            )
        ax.legend(loc="best")

    fig.tight_layout()
    return fig


def build_cluster_tables(
    *,
    report: EmbeddingClusterReport,
    is_truck_follower: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    follower_mask = np.asarray(is_truck_follower, dtype=bool).reshape(-1)
    if follower_mask.shape[0] != report.tokens.shape[0]:
        raise ValueError(
            "is_truck_follower length mismatch: "
            f"{follower_mask.shape[0]} vs {report.tokens.shape[0]}"
        )

    def _group_table(group: ClusterGroup) -> pd.DataFrame:
        if group.sample_indices.size == 0:
            return pd.DataFrame(
                columns=[
                    "cluster",
                    "count",
                    "self_is_truck",
                    "self_is_pc",
                    "truck_ratio",
                    "pc_ratio",
                ]
            )

        sample_follower = follower_mask[group.sample_indices]
        rows: list[dict[str, float | int]] = []
        truck_total = float(np.sum(sample_follower))
        pc_total = float(sample_follower.size - np.sum(sample_follower))

        for cluster_id in range(group.num_clusters):
            cluster_mask = group.labels == cluster_id
            self_is_truck = int(np.sum(sample_follower[cluster_mask]))
            self_is_pc = int(np.sum(cluster_mask) - self_is_truck)
            rows.append(
                {
                    "cluster": int(cluster_id),
                    "count": int(np.sum(cluster_mask)),
                    "self_is_truck": self_is_truck,
                    "self_is_pc": self_is_pc,
                    "truck_ratio": float(self_is_truck / truck_total) if truck_total > 0 else float("nan"),
                    "pc_ratio": float(self_is_pc / pc_total) if pc_total > 0 else float("nan"),
                }
            )
        return pd.DataFrame(rows)

    return _group_table(report.truck), _group_table(report.notruck)


def plot_cluster_composition(table: pd.DataFrame, *, title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 6))
    if table.empty:
        ax.text(0.5, 0.5, "No samples", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        fig.tight_layout()
        return fig

    clusters = table["cluster"].astype(int).tolist()
    colors = plt.get_cmap("tab10").colors

    x_positions = [0, 1]
    bottom_truck = 0.0
    bottom_pc = 0.0

    for idx, cluster_id in enumerate(clusters):
        row = table.loc[table["cluster"] == cluster_id].iloc[0]
        truck_ratio = float(row["truck_ratio"]) if np.isfinite(float(row["truck_ratio"])) else 0.0
        pc_ratio = float(row["pc_ratio"]) if np.isfinite(float(row["pc_ratio"])) else 0.0
        color = colors[idx % len(colors)]

        ax.bar(x_positions[0], truck_ratio, bottom=bottom_truck, width=0.5, color=color, label=f"C{cluster_id}")
        ax.bar(x_positions[1], pc_ratio, bottom=bottom_pc, width=0.5, color=color)
        bottom_truck += truck_ratio
        bottom_pc += pc_ratio

    truck_total = int(table["self_is_truck"].sum())
    pc_total = int(table["self_is_pc"].sum())

    ax.set_xticks(x_positions)
    ax.set_xticklabels([f"Truck (n={truck_total})", f"Passenger Car (n={pc_total})"])
    ax.set_ylabel("Proportion")
    ax.set_ylim(0.0, 1.0)
    ax.set_title(title)
    ax.legend(title="Cluster", loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=4, frameon=False)
    fig.tight_layout()
    return fig


__all__ = [
    "ClusterGroup",
    "ClusterResult",
    "EmbeddingClusterReport",
    "EmbeddingClusterProtocol",
    "cluster_embeddings_kmeans",
    "cluster_embeddings_gmm",
    "cluster_embeddings_hdbscan",
    "cluster_style_embeddings",
    "plot_embedding_clusters",
    "build_cluster_tables",
    "plot_cluster_composition",
]
