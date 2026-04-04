"""
Style experiment pipeline entrypoint.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from typing import Mapping

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from .schema import CFNAMES as CF, TensorNames
from .utils.logger import logger
from .exps import get_style_configs, load_common_config
from .exps import resolve_common_runtime
from .exps.agent import Agent
from .exps.datahandle.feat_extractor import reaction_time, time_headway
from .exps.datahandle.stylebuilder import build_style_tokens_from_datapack
from .exps.models.model_loader import load_state_if_available
from .exps.models.stylecf import EmbeddingStyleTransformer, StyleTransformer, style_embed_mask, style_embed_update_train_series
from .exps.style.embedding_report import (
    build_cluster_tables,
    cluster_style_embeddings,
    plot_cluster_composition,
    plot_embedding_clusters,
)
from .exps.style.quantitive_summary import (
    _build_cluster_style_dataframe,
    _log_cluster_style_summary,
    _summarize_cluster_style,
)
from .exps.style.transfer_report import (
    plot_cluster_vs_embedding_styles,
    plot_follower_type_styles,
    run_cluster_transfer_grid,
    run_follower_type_substitution,
    summarize_cluster_transfer,
    summarize_follower_transfer,
)
from .exps.test.data_bundle import load_eval_bundle
from .exps.test.testing_utils import build_group_input, build_test_traj_builder, concat_time, style_embed_pred_func
from .exps.utils import SampleDataPack
from .utils.config_utils import _require_option


def _with_timestamp_subdir(base_dir: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return base_dir / timestamp


STYLE_START_TIME = 0


def _load_runtime_configs() -> tuple[dict[str, object], dict[str, object]]:
    common_cfg = resolve_common_runtime(load_common_config()["config"])
    style_cfg = get_style_configs()
    return common_cfg, style_cfg


@dataclass(frozen=True)
class StyleCaseResult:
    prediction: object
    metrics: dict
    data: object


def analyze(
    pred_self: torch.Tensor,
    true_leader: torch.Tensor,
    start_time: int,
    lead_len: float,
    dt: float,
) -> dict[str, float]:
    start = max(0, min(int(start_time), int(pred_self.shape[0]) - 1))
    pred_np = pred_self.detach().cpu().numpy()
    lead_np = true_leader.detach().cpu().numpy()

    time = np.arange(pred_np.shape[0], dtype=np.float32) * float(dt)
    react = reaction_time(
        leader_v=lead_np[:, 1],
        self_v=pred_np[:, 1],
        time=time,
    )[start:]
    thw = time_headway(
        spacing=lead_np[:, 0] - pred_np[:, 0] - float(lead_len),
        self_v=pred_np[:, 1],
    )[start:]

    react = react[np.isfinite(react)]
    thw = thw[np.isfinite(thw)]
    return {
        "thw": float(np.mean(thw)) if thw.size > 0 else float("nan"),
        "react": float(np.mean(react)) if react.size > 0 else float("nan"),
    }




def _build_style_model(
    device: torch.device,
    style_pipeline_config: Mapping[str, object],
    style_data_config: Mapping[str, object],
) -> StyleTransformer:
    t0 = perf_counter()
    path = style_pipeline_config["style_state_path"]
    if path in (None, "", ...):
        raise ValueError("Missing style_pipeline_config['style_state_path'].")
    path = str(path)
    model = StyleTransformer(style_data_config)
    loaded = load_state_if_available(model, path, device, strict=True).eval()
    logger.info(
        "Style model ready | "
        f"path={path} device={device} training={loaded.training} elapsed={perf_counter() - t0:.2f}s"
    )
    return loaded


def _filter_by_cluster(
    data: SampleDataPack,
    labels: np.ndarray | None = None,
    target: int | None = None,
    is_truck: np.ndarray | None = None,
    keep_truck: bool = False,
    *,
    cluster_labels: np.ndarray | None = None,
    target_cluster: int | None = None,
) -> SampleDataPack:
    label_arr = cluster_labels if cluster_labels is not None else labels
    target_value = target_cluster if target_cluster is not None else target
    if label_arr is None or target_value is None:
        raise ValueError("Both cluster labels and target cluster must be provided.")
    if is_truck is None:
        mask = label_arr == int(target_value)
    else:
        mask = (is_truck == keep_truck) & (label_arr == int(target_value))
    return data.select_rows(np.flatnonzero(mask))


def _build_style_tokens(
    options: Mapping[str, object],
    d_style,
    d_test,
    model,
    scalers,
    device,
    style_data_config: Mapping[str, object],
) -> np.ndarray:
    t0 = perf_counter()
    feat_names = list(style_data_config["x_groups"]["style"]["features"])
    token_source = str(_require_option(options, "style_token_source")).lower()
    token_batch_size = int(_require_option(options, "style_token_batch_size"))
    token_log_every_batches = int(_require_option(options, "style_token_log_every_batches"))
    kwargs = dict(
        feature_names=feat_names,
        embedder=model.embedder,
        scaler=scalers.get("style"),
        device=device,
        batch_size=token_batch_size,
        log_every_batches=token_log_every_batches,
    )
    logger.info(
        "Build style tokens start | "
        f"source={token_source} d_style_shape={tuple(d_style.data.shape)} "
        f"d_test_shape={tuple(d_test.data.shape)} features={feat_names} "
        f"batch_size={token_batch_size} log_every_batches={token_log_every_batches}"
    )

    if token_source in {"full_length", "full", "all"}:
        full_seconds = float(d_style.data.shape[1]) * float(d_style.dt)
        logger.info(f"Build style tokens window | full_length seconds={full_seconds:.2f}")
        tokens = build_style_tokens_from_datapack(datapack=d_style, seconds=full_seconds, **kwargs)
    elif token_source == "style_window_head":
        seconds = float(_require_option(options, "style_token_seconds"))
        logger.info(f"Build style tokens window | style_window_head seconds={seconds:.2f}")
        tokens = build_style_tokens_from_datapack(datapack=d_style, seconds=seconds, **kwargs)
    elif token_source == "nearby_before_start":
        anchor = STYLE_START_TIME
        logger.info(
            "Build style tokens window | "
            f"nearby_before_start anchor={anchor} before_seconds={_require_option(options, 'style_window_before_seconds')}"
        )
        tokens = build_style_tokens_from_datapack(
            datapack=d_test,
            seconds=float(_require_option(options, "style_token_seconds")),
            end_step=anchor,
            window_before_seconds=_require_option(options, "style_window_before_seconds"),
            **kwargs,
        )
    else:
        raise ValueError(
            f"Unsupported style_token_source='{token_source}'. "
            "Expected one of: full_length, style_window_head, nearby_before_start."
        )
    out = tokens.detach().cpu().numpy().astype(np.float32)
    logger.info(f"Build style tokens done | shape={out.shape} elapsed={perf_counter() - t0:.2f}s")
    return out


class StyleNotebookPipeline:
    def __init__(self, agent, data_config, scalers, pred_func, mask, device):
        self.agent, self.data_config, self.scalers, self.pred_func, self.mask, self.device = agent, data_config, scalers, pred_func, mask, device

    def run_sample(self, data, embedding, sample, start_time=60, lead_len=0.0, distance_offset=0.0) -> StyleCaseResult:
        sample_data = data.select_rows(np.array([int(sample)]))
        batch = build_group_input(
            sample_data,
            0,
            self.data_config["x_groups"],
            self.scalers,
            ("enc_x", "dec_x"),
            self.device,
        )
        batch["style_embed"] = torch.tensor(
            np.asarray(embedding[0], dtype=np.float32),
            dtype=torch.float32,
            device=self.device,
        ).rename(TensorNames.F)

        build_traj = build_test_traj_builder(sample_data, self.device)
        self_traj, leader_traj = build_traj(0)
        pred_self = self.agent.predict(
            batch,
            self_traj,
            leader_traj,
            pred_func=self.pred_func,
            mask=self.mask,
        )
        res = SimpleNamespace(pred_self=pred_self, true_leader=leader_traj)
        num_steps = min(int(res.pred_self.shape[0]), int(res.true_leader.shape[0]))
        if num_steps <= 0:
            raise ValueError("Empty rollout result.")
        eval_start = max(0, min(int(start_time), num_steps - 1))
        metrics = analyze(
            res.pred_self[:num_steps],
            res.true_leader[:num_steps],
            eval_start,
            float(lead_len),
            float(sample_data.dt),
        )
        return StyleCaseResult(prediction=res, metrics=metrics, data=sample_data)

    def run_cluster_transfer(self, data, cluster_labels, cluster, compare_cluster, centroids, sample, **kwargs) -> StyleCaseResult:
        filtered = _filter_by_cluster(data, cluster_labels, cluster, kwargs.get("is_truck"), kwargs.get("keep_truck", False))
        if kwargs.get("follower_length_override") and CF.SELF_L in filtered.names:
            filtered.replace_col(np.ones_like(filtered.data[:,:,0]) * kwargs["follower_length_override"], CF.SELF_L)
        return self.run_sample(filtered, centroids[compare_cluster][None, :], sample, kwargs.get("start_time", 60), distance_offset=kwargs.get("distance_offset", 0.0))


def run_style_pipeline_from_config():
    total_t0 = perf_counter()
    common_cfg, style_cfg = _load_runtime_configs()
    style_data_config = common_cfg["style_data_config"]
    style_pipeline_config = style_cfg["style_pipeline_config"]

    opt = style_pipeline_config
    output_dir = _with_timestamp_subdir(Path(_require_option(opt, "output_dir")))
    device = torch.device(_require_option(opt, "device"))
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Style pipeline start")

    t0 = perf_counter()
    bundle = load_eval_bundle(
        test_config=style_pipeline_config,
        style_data_config=style_data_config,
        lstm_data_config=common_cfg["lstm_data_config"],
        filter_names=common_cfg["filter_names"],
        data_filter_config=common_cfg["data_filter_config"],
        head=opt.get("head"),
        use_split_windows=bool(_require_option(opt, "use_split_windows")),
        use_full_window=bool(_require_option(opt, "use_full_window")),
        split_log_prefix="Style",
    )
    d_test, scalers = bundle.d_test, bundle.style_scalers
    logger.info(
        "Eval bundle ready | "
        f"d_style_shape={tuple(bundle.d_style.data.shape)} d_test_shape={tuple(d_test.data.shape)} "
        f"scalers={list(scalers.keys())} elapsed={perf_counter() - t0:.2f}s"
    )
    
    model = _build_style_model(device, style_pipeline_config, style_data_config)
    tokens = _build_style_tokens(opt, bundle.d_style, d_test, model, scalers, device, style_data_config)
    cluster_protocol = _require_option(opt, "cluster")

    t0 = perf_counter()
    lead_length_threshold = float(_require_option(opt, "lead_length_threshold"))
    report = cluster_style_embeddings(
        tokens=tokens,
        is_truck_leader=np.asarray(d_test[:, 0, CF.LEAD_L] >= lead_length_threshold),
        lead_length_threshold=lead_length_threshold,
        truck_clusters=int(_require_option(opt, "truck_clusters")),
        notruck_clusters=int(_require_option(opt, "notruck_clusters")),
        random_seed=int(_require_option(opt, "random_seed")),
        cluster_protocol=cluster_protocol,
    )
    logger.info(
        "Embedding clustering done | "
        f"method={cluster_protocol['method']} "
        f"truck_samples={int(report.truck.sample_indices.size)} truck_clusters={report.truck.num_clusters} "
        f"notruck_samples={int(report.notruck.sample_indices.size)} notruck_clusters={report.notruck.num_clusters} "
        f"elapsed={perf_counter() - t0:.2f}s"
    )

    t0 = perf_counter()
    np.save(output_dir / "style_embeddings.npy", report.tokens)
    truck_table, notruck_table = build_cluster_tables(report=report, is_truck_follower=np.asarray(d_test[:, 0, CF.SELF_L] >= lead_length_threshold))
    truck_table.to_csv(output_dir / "truck_cluster_stats.csv", index=False)
    notruck_table.to_csv(output_dir / "notruck_cluster_stats.csv", index=False)
    logger.info(
        "Cluster composition tables saved | "
        f"truck_rows={len(truck_table)} notruck_rows={len(notruck_table)} elapsed={perf_counter() - t0:.2f}s"
    )

    t0 = perf_counter()
    truck_style_df = _build_cluster_style_dataframe(
        data=d_test,
        sample_indices=report.truck.sample_indices,
        labels=report.truck.labels,
        lead_length_threshold=lead_length_threshold,
        progress_desc="Truck style summary",
    )
    notruck_style_df = _build_cluster_style_dataframe(
        data=d_test,
        sample_indices=report.notruck.sample_indices,
        labels=report.notruck.labels,
        lead_length_threshold=lead_length_threshold,
        progress_desc="Notruck style summary",
    )
    truck_style_summary = _summarize_cluster_style(truck_style_df)
    notruck_style_summary = _summarize_cluster_style(notruck_style_df)
    truck_style_df.to_csv(output_dir / "truck_cluster_style_samples.csv", index=False)
    notruck_style_df.to_csv(output_dir / "notruck_cluster_style_samples.csv", index=False)
    truck_style_summary.to_csv(output_dir / "truck_cluster_style_summary.csv", index=False)
    notruck_style_summary.to_csv(output_dir / "notruck_cluster_style_summary.csv", index=False)
    _log_cluster_style_summary(truck_style_summary, label="Truck")
    _log_cluster_style_summary(notruck_style_summary, label="Notruck")
    logger.info(
        "Cluster style summaries saved | "
        f"truck_samples={len(truck_style_df)} notruck_samples={len(notruck_style_df)} "
        f"elapsed={perf_counter() - t0:.2f}s"
    )

    if bool(_require_option(opt, "enable_embedding_plots")):
        t0 = perf_counter()
        plot_embedding_clusters(report).savefig(output_dir / "embedding_clusters_pca.png", dpi=180)
        plot_cluster_composition(truck_table, title="Truck Clusters").savefig(output_dir / "truck_cluster_comp.png", dpi=180)
        logger.info(f"Embedding plots saved | elapsed={perf_counter() - t0:.2f}s")

    if bool(_require_option(opt, "enable_cluster_transfer")) and report.notruck.num_clusters > 0:
        t0 = perf_counter()
        embed_model = EmbeddingStyleTransformer(model).to(device).eval()
        agent = Agent(embed_model, d_test.dt, int(style_data_config["pred_len"]), int(style_data_config["seq_len"]), scalers, start_timestep=0)
        agent._update_train_series = style_embed_update_train_series(agent, style_data_config)
        agent._concat = concat_time
        
        runner = StyleNotebookPipeline(agent=agent, data_config=style_data_config, scalers=scalers, pred_func=style_embed_pred_func, mask=style_embed_mask(style_data_config), device=device)
        is_truck = np.asarray(d_test[:, 0, CF.LEAD_L] >= lead_length_threshold)
        d_notruck = d_test.select_rows(np.flatnonzero(~is_truck))
        logger.info(
            "Cluster transfer start | "
            f"notruck_samples={int(d_notruck.data.shape[0])} clusters={report.notruck.num_clusters}"
        )
        res = run_cluster_transfer_grid(
            runner=runner,
            data=d_notruck,
            filter_by_cluster=_filter_by_cluster,
            cluster_labels=report.notruck.labels,
            centroids=report.notruck.centroids,
            start_time=STYLE_START_TIME,
            is_truck=None,
            keep_truck=False,
            max_samples_per_cluster=int(_require_option(opt, "transfer_max_samples_per_cluster")),
            random_seed=int(_require_option(opt, "random_seed")),
            distance_offset=float(_require_option(opt, "distance_offset")),
            show_progress=True,
        )
        transfer_df = summarize_cluster_transfer(res)
        transfer_df.to_csv(output_dir / "cluster_transfer.csv", index=False)
        plot_cluster_vs_embedding_styles(
            results=res,
            num_clusters=report.notruck.num_clusters,
            xlim=_require_option(opt, "kde_xlim"),
            show_progress=True,
        ).savefig(output_dir / "cluster_transfer_kde.png", dpi=180)
        logger.info(
            "Cluster transfer done | "
            f"rows={len(transfer_df)} elapsed={perf_counter() - t0:.2f}s"
        )

    logger.info(f"Style pipeline done | elapsed={perf_counter() - total_t0:.2f}s")
    return {"output_dir": str(output_dir)}


def main():
    res = run_style_pipeline_from_config()
    print(f"Style pipeline finished. Results in: {res['output_dir']}")

if __name__ == "__main__":
    main()

__all__ = ["StyleNotebookPipeline", "run_style_pipeline_from_config", "main"]
