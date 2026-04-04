from __future__ import annotations

import logging
from time import perf_counter

import numpy as np
import pandas as pd

from ...schema import CFNAMES as CF
from ..utils.utils import SampleDataPack

logger = logging.getLogger(__name__)


_CLUSTER_STYLE_COLUMNS = [
    "sample_id",
    "cluster",
    "reaction_time",
    "time_headway",
    "spacing",
    "self_v",
    "self_len",
    "lead_len",
    "avg_jerk",
    "std_react",
    "std_thw",
    "self_is_truck",
    "self_is_pc",
    "lead_is_truck",
    "speed_class",
]


_CLUSTER_SUMMARY_COLUMNS = [
    "cluster",
    "count",
    "reaction_time_mean",
    "reaction_time_std",
    "time_headway_mean",
    "time_headway_std",
    "spacing_mean",
    "spacing_std",
    "avg_jerk_mean",
    "avg_jerk_std",
    "std_react_mean",
    "std_react_std",
    "std_thw_mean",
    "std_thw_std",
]



def _build_cluster_style_dataframe(
    *,
    data: SampleDataPack,
    sample_indices: np.ndarray,
    labels: np.ndarray,
    lead_length_threshold: float,
    progress_desc: str,
) -> pd.DataFrame:
    t0 = perf_counter()

    sample_ids = np.asarray(sample_indices, dtype=np.int64).reshape(-1)
    cluster_labels = np.asarray(labels, dtype=np.int64).reshape(-1)

    assert sample_ids.ndim == 1
    assert cluster_labels.ndim == 1
    assert sample_ids.size > 0
    assert sample_ids.size == cluster_labels.size
    assert np.all(sample_ids >= 0)
    assert float(lead_length_threshold) > 0.0
    assert isinstance(progress_desc, str)

    required_features = [CF.REACT, CF.THW, CF.DELTA_X, CF.SELF_V, CF.SELF_A, CF.SELF_L, CF.LEAD_L]
    missing = [name for name in required_features if name not in data.names]
    assert not missing, f"Missing required features: {missing}"

    values = np.asarray(data.data)
    assert values.ndim == 3
    num_samples, num_steps, _ = values.shape
    assert num_samples > 0
    assert num_steps >= 2, "Need at least 2 timesteps to compute jerk."
    assert int(sample_ids.max()) < num_samples

    logger.info(
        f"Build cluster style dataframe start | desc={progress_desc} "
        f"samples={int(sample_ids.size)}"
    )

    sample_block = values[sample_ids]
    assert sample_block.shape[0] == sample_ids.size
    assert sample_block.shape[1] == num_steps

    react = sample_block[:, :, data.names[CF.REACT]]
    thw = sample_block[:, :, data.names[CF.THW]]
    delta_x = sample_block[:, :, data.names[CF.DELTA_X]]
    self_v = sample_block[:, :, data.names[CF.SELF_V]]
    self_a = sample_block[:, :, data.names[CF.SELF_A]]
    self_len = sample_block[:, 0, data.names[CF.SELF_L]]
    lead_len = sample_block[:, 0, data.names[CF.LEAD_L]]

    assert np.isfinite(react).all()
    assert np.isfinite(thw).all()
    assert np.isfinite(delta_x).all()
    assert np.isfinite(self_v).all()
    assert np.isfinite(self_a).all()
    assert np.isfinite(self_len).all()
    assert np.isfinite(lead_len).all()

    spacing = delta_x - lead_len[:, None]
    jerk = np.abs(np.gradient(self_a, axis=1))

    out = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "cluster": cluster_labels,
            "reaction_time": react.mean(axis=1),
            "time_headway": thw.mean(axis=1),
            "spacing": spacing.mean(axis=1),
            "self_v": self_v.mean(axis=1),
            "self_len": self_len,
            "lead_len": lead_len,
            "avg_jerk": jerk.mean(axis=1),
            "std_react": react.std(axis=1),
            "std_thw": thw.std(axis=1),
            "self_is_truck": (self_len >= float(lead_length_threshold)).astype(np.int64),
            "self_is_pc": (self_len < float(lead_length_threshold)).astype(np.int64),
            "lead_is_truck": (lead_len >= float(lead_length_threshold)).astype(np.int64),
            "speed_class": (self_v.mean(axis=1) >= 16.7).astype(np.int64),
        }
    )

    assert list(out.columns) == _CLUSTER_STYLE_COLUMNS
    assert len(out) == sample_ids.size
    assert out.notna().all().all()

    logger.info(
        f"Build cluster style dataframe done | desc={progress_desc} "
        f"rows={len(out)} elapsed={perf_counter() - t0:.2f}s"
    )
    return out



def _summarize_cluster_style(df: pd.DataFrame) -> pd.DataFrame:
    assert isinstance(df, pd.DataFrame)
    assert not df.empty

    required_cols = [
        "cluster",
        "reaction_time",
        "time_headway",
        "spacing",
        "avg_jerk",
        "std_react",
        "std_thw",
    ]
    missing = [col for col in required_cols if col not in df.columns]
    assert not missing, f"Missing required columns: {missing}"
    assert df[required_cols].notna().all().all()

    metric_cols = ["reaction_time", "time_headway", "spacing", "avg_jerk", "std_react", "std_thw"]
    grouped = df.groupby("cluster", sort=True)
    means = grouped[metric_cols].mean().rename(columns=lambda col: f"{col}_mean")
    stds = grouped[metric_cols].std(ddof=0).rename(columns=lambda col: f"{col}_std")
    counts = grouped.size().astype(int).rename("count")

    out = pd.concat([means, stds, counts], axis=1).reset_index()
    out = out[[
        "cluster",
        "count",
        "reaction_time_mean",
        "reaction_time_std",
        "time_headway_mean",
        "time_headway_std",
        "spacing_mean",
        "spacing_std",
        "avg_jerk_mean",
        "avg_jerk_std",
        "std_react_mean",
        "std_react_std",
        "std_thw_mean",
        "std_thw_std",
    ]]

    assert list(out.columns) == _CLUSTER_SUMMARY_COLUMNS
    assert out["count"].gt(0).all()
    return out



def _log_cluster_style_summary(df: pd.DataFrame, *, label: str) -> None:
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert isinstance(label, str) and label

    missing = [col for col in _CLUSTER_SUMMARY_COLUMNS if col not in df.columns]
    assert not missing, f"Missing required columns: {missing}"

    for row in df.itertuples(index=False):
        logger.info(
            f"{label} cluster summary | "
            f"cluster={int(row.cluster)} count={int(row.count)} "
            f"thw_mean={float(row.time_headway_mean):.4f} "
            f"react_mean={float(row.reaction_time_mean):.4f} "
            f"spacing_mean={float(row.spacing_mean):.4f} "
            f"std_thw_mean={float(row.std_thw_mean):.4f} "
            f"std_react_mean={float(row.std_react_mean):.4f}"
        )
