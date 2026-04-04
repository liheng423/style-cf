from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Mapping, cast

import matplotlib.pyplot as plt
import torch

from .exps.configs import (
    DEFAULT_STYLE_WINDOW,
    DEFAULT_TEST_WINDOW,
    data_filter_config,
    filter_names,
    idm_calibration_config,
    lstm_data_config,
    lstm_model_config,
    test_config,
    style_data_config,
)
from .exps.models.testing_builders import BuilderContext, get_model_builders
from .exps.test.data_bundle import build_style_datapack, split_eval_windows
from .exps.test.model_tester import (
    ModelEvalResult,
    evaluate_rollout,
    save_result_bundle,
    summarize_results,
)
from .exps.test.scaler_config import load_test_scalers
from .exps.test.testing_utils import (
    build_test_traj_builder,
)
from .exps.utils.datapack import SampleDataPack
from .exps.utils.split_io import load_split_indices
from .exps.test.visualizer import plot_error_evolution, plot_metric_histograms
from .utils.rawdata_loader import load_datapack
from .utils.logger import logger


def _with_timestamp_subdir(base_dir: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return base_dir / timestamp


def _resolve_rollout_start_time(
    requested_start: int,
    total_steps: int,
    historic_step: int,
    rollout_step: int,
) -> int:
    start = max(0, int(requested_start))
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if historic_step < 0:
        raise ValueError("historic_step must be non-negative")
    if rollout_step <= 0:
        raise ValueError("rollout_step must be positive")
    if start >= total_steps:
        raise ValueError(f"start_time={start} is out of range for sequence length {total_steps}")

    remaining = total_steps - start - historic_step
    if remaining < 0:
        raise ValueError(
            f"start_time={start} leaves fewer than historic_step={historic_step} steps "
            f"for sequence length {total_steps}"
        )

    remainder = remaining % rollout_step
    if remainder == 0:
        return start

    adjusted = start + remainder
    remaining_after_adjust = total_steps - adjusted - historic_step
    if remaining_after_adjust < 0:
        raise ValueError(
            "Cannot align rollout with current start_time/historic_step/rollout_step settings."
        )
    return adjusted


@dataclass
class TestingOptions:
    head: int | None
    style_window: tuple[int, int]
    test_window: tuple[int, int]
    use_split_windows: bool
    start_time: int
    style_token_seconds: float
    style_window_before_seconds: tuple[float, float] | None
    style_token_mode: str
    style_token_source: str
    output_dir: Path
    enabled_models: tuple[str, ...]
    save_results: bool
    plot_results: bool
    use_saved_split: bool = False
    split_index_path: str | None = None
    split_strict: bool = False


@dataclass
class EvalBundle:
    d_style: object
    d_test: object
    style_scalers: dict[str, object]
    transformer_scalers: dict[str, object]
    lstm_scalers: dict[str, object]


def _apply_saved_test_split(
    d_full,
    split_index_path: str | None,
    strict: bool,
):
    if not split_index_path:
        return d_full

    path = Path(str(split_index_path))
    if not path.exists():
        msg = f"Saved split file not found: {path}"
        if strict:
            raise FileNotFoundError(msg)
        logger.warning(msg + " | fallback to full filtered dataset.")
        return d_full

    payload = load_split_indices(path)
    if "test_idx" not in payload:
        raise KeyError(f"Split file missing 'test_idx': {path}")
    test_idx = payload["test_idx"]

    total = int(d_full.data.shape[0])
    if test_idx.ndim != 1:
        raise ValueError(f"test_idx must be 1D, got shape {tuple(test_idx.shape)}")
    if test_idx.size == 0:
        raise ValueError(f"test_idx is empty in split file: {path}")
    if (test_idx < 0).any() or (test_idx >= total).any():
        raise ValueError(
            f"test_idx out of range for filtered dataset size {total}. "
            "Ensure testing uses the same filtering config as training."
        )

    data = d_full.data[test_idx]
    d_test = SampleDataPack(
        data=data,
        name_dict=d_full.names.copy(),
        rise=d_full.rise,
        kph=d_full.kph,
        kilo_norm=d_full.kilo_norm,
        dt=d_full.dt,
    )
    logger.info(
        f"Applied saved test split from {path} | "
        f"test_samples={int(test_idx.size)} filtered_total={total}"
    )
    return d_test


def _resolve_device() -> torch.device:
    device = test_config.get("device")
    if isinstance(device, str):
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _dataset(head: int | None = None):
    rawdata_name = test_config.get("rawdata_config", test_config.get("datacfg"))
    if rawdata_name in (None, "", ...):
        raise ValueError("Missing test_config['rawdata_config'] (legacy key 'datacfg' is also empty).")
    d, _, _ = load_datapack(str(rawdata_name))
    return d.head(head) if head is not None else d


def _build_options() -> TestingOptions:
    head_env = os.environ.get("TEST_HEAD")
    head = int(head_env) if head_env else None

    style_window = tuple(test_config.get("style_window", DEFAULT_STYLE_WINDOW))
    test_window = tuple(test_config.get("test_window", DEFAULT_TEST_WINDOW))

    style_token_seconds = float(test_config.get("style_token_seconds", 6.0))
    raw_before = test_config.get("style_window_before_seconds")
    style_window_before_seconds: tuple[float, float] | None
    if isinstance(raw_before, (list, tuple)) and len(raw_before) == 2:
        style_window_before_seconds = (float(raw_before[0]), float(raw_before[1]))
    else:
        style_window_before_seconds = None
    style_token_mode = str(test_config.get("style_token_mode", "per_sample"))
    style_token_source = str(test_config.get("style_token_source", "style_window_head"))
    use_split_windows = bool(test_config.get("use_split_windows", True))
    use_saved_split = bool(test_config.get("use_saved_split", False))
    split_index_path_raw = test_config.get("split_index_path")
    split_index_path = None if split_index_path_raw in (None, "", ...) else str(split_index_path_raw)
    split_strict = bool(test_config.get("saved_split_strict", False))
    start_time = int(test_config.get("start_time", 60))

    models_cfg = test_config.get("enabled_models")
    if isinstance(models_cfg, (list, tuple)) and models_cfg:
        enabled_models = tuple(str(x).lower() for x in models_cfg)
    else:
        enabled_models = ["stylecf", "transformer", "idm"]
        if "lstm_agent" in test_config:
            enabled_models.append("lstm")
        enabled_models = tuple(enabled_models)

    output_dir = _with_timestamp_subdir(Path(str(test_config.get("output_dir", "models/test_results"))))

    plot_env = os.environ.get("TEST_PLOT")
    if plot_env is not None:
        plot_results = plot_env.strip() in {"1", "true", "True"}
    else:
        plot_results = bool(test_config.get("plot_results", False))

    return TestingOptions(
        head=head,
        style_window=cast(tuple[int, int], style_window),
        test_window=cast(tuple[int, int], test_window),
        use_split_windows=use_split_windows,
        start_time=start_time,
        style_token_seconds=style_token_seconds,
        style_window_before_seconds=style_window_before_seconds,
        style_token_mode=style_token_mode,
        style_token_source=style_token_source,
        use_saved_split=use_saved_split,
        split_index_path=split_index_path,
        split_strict=split_strict,
        output_dir=output_dir,
        enabled_models=cast(tuple[str, ...], enabled_models),
        save_results=bool(test_config.get("save_results", True)),
        plot_results=plot_results,
    )


def _build_eval_bundle(
    head: int | None = None,
    style_window: tuple[int, int] = DEFAULT_STYLE_WINDOW,
    test_window: tuple[int, int] = DEFAULT_TEST_WINDOW,
    use_split_windows: bool = True,
    use_saved_split: bool = False,
    split_index_path: str | None = None,
    split_strict: bool = False,
) -> EvalBundle:
    raw_data = _dataset(head=head)
    d_full = build_style_datapack(raw_data, filter_names, data_filter_config)
    if use_saved_split:
        d_full = _apply_saved_test_split(
            d_full,
            split_index_path=split_index_path,
            strict=split_strict,
        )
    if use_split_windows:
        d_style, d_test = split_eval_windows(d_full, style_window, test_window)
    else:
        total_steps = int(d_full.data.shape[1])
        test_start, test_end = test_window
        if not (0 <= test_start < test_end <= total_steps):
            raise ValueError(
                f"test_window={test_window} is invalid for sequence length {total_steps}. "
                "Update test_config['test_window']."
            )
        d_test = d_full.split_by_time_windows_list([test_window])[0]
        # Keep style datapack available for compatibility with style_window_head mode.
        d_style = d_test

    style_scalers, transformer_scalers, lstm_scalers = load_test_scalers(
        test_config,
        style_data_config["x_groups"],
    )

    return EvalBundle(
        d_style=d_style,
        d_test=d_test,
        style_scalers=style_scalers,
        transformer_scalers=transformer_scalers,
        lstm_scalers=lstm_scalers,
    )


def run_testing(options: TestingOptions | None = None) -> dict[str, ModelEvalResult]:
    options = options or _build_options()
    device = _resolve_device()

    eval_bundle = _build_eval_bundle(
        head=options.head,
        style_window=options.style_window,
        test_window=options.test_window,
        use_split_windows=options.use_split_windows,
        use_saved_split=options.use_saved_split,
        split_index_path=options.split_index_path,
        split_strict=options.split_strict,
    )
    d_style = eval_bundle.d_style
    d_test = eval_bundle.d_test

    num_samples = d_test.data.shape[0]
    build_traj = build_test_traj_builder(d_test, device)

    style_token_anchor_step = _resolve_rollout_start_time(
        requested_start=options.start_time,
        total_steps=int(d_test.data.shape[1]),
        historic_step=int(style_data_config["seq_len"]),
        rollout_step=int(style_data_config["pred_len"]),
    )

    results: dict[str, ModelEvalResult] = {}
    context = BuilderContext(
        device=device,
        d_style=d_style,
        d_test=d_test,
        style_scalers=eval_bundle.style_scalers,
        transformer_scalers=eval_bundle.transformer_scalers,
        lstm_scalers=eval_bundle.lstm_scalers,
        style_token_seconds=options.style_token_seconds,
        style_window_before_seconds=options.style_window_before_seconds,
        style_token_mode=options.style_token_mode,
        style_token_source=options.style_token_source,
        style_token_anchor_step=style_token_anchor_step,
        test_config=test_config,
        style_data_config=style_data_config,
        lstm_data_config=lstm_data_config,
        lstm_model_config=lstm_model_config,
        idm_calibration_config=idm_calibration_config,
    )

    builders = get_model_builders()
    for model_name in options.enabled_models:
        builder = builders.get(model_name)
        if builder is None:
            print(f"[testing] unsupported model '{model_name}', skipped.")
            continue

        runner = builder.build(context)
        total_steps = int(d_test.data.shape[1])
        effective_start_time = _resolve_rollout_start_time(
            requested_start=options.start_time,
            total_steps=total_steps,
            historic_step=int(runner.agent.historic_step),
            rollout_step=int(runner.agent.rollout_step),
        )
        runner.agent.start_step = effective_start_time
        if effective_start_time != options.start_time:
            print(
                "[testing] adjusted start_time "
                f"{options.start_time} -> {effective_start_time} for model '{runner.model_name}' "
                "to align rollout windows."
            )

        results[runner.model_name] = evaluate_rollout(
            model_name=runner.model_name,
            agent=runner.agent,
            num_samples=num_samples,
            build_x=runner.build_x,
            build_traj=build_traj,
            pred_func=runner.pred_func,
            mask=runner.mask,
            start_time=effective_start_time,
        )

    return results


def save_outputs(results: Mapping[str, ModelEvalResult], options: TestingOptions, dt: float) -> None:
    options.output_dir.mkdir(parents=True, exist_ok=True)

    summary_path, per_model_paths = save_result_bundle(
        results=results,
        output_dir=options.output_dir,
        save_errors=True,
        merge_with_existing_summary=True,
    )
    print(f"[testing] summary saved: {summary_path}")

    for metric_path, error_path in per_model_paths.values():
        print(f"[testing] metrics saved: {metric_path}")
        if error_path is not None:
            print(f"[testing] errors saved: {error_path}")

    if options.plot_results and results:
        fig_hist = plot_metric_histograms(results)
        hist_path = options.output_dir / "metric_histograms.png"
        fig_hist.savefig(hist_path, dpi=180)
        plt.close(fig_hist)
        print(f"[testing] figure saved: {hist_path}")

        fig_evo = plot_error_evolution(results, dt=dt)
        evo_path = options.output_dir / "error_evolution.png"
        fig_evo.savefig(evo_path, dpi=180)
        plt.close(fig_evo)
        print(f"[testing] figure saved: {evo_path}")


def run_from_config(options: TestingOptions | None = None) -> dict[str, ModelEvalResult]:
    options = options or _build_options()
    results = run_testing(options)

    if not results:
        print("[testing] no models were evaluated. Check test_config['enabled_models'].")
        return {}

    summary_df = summarize_results(results)
    print("[testing] mean metrics")
    print(summary_df.to_string(index=False))

    if options.save_results:
        save_outputs(results, options, dt=0.1)

    return results


def main() -> int:
    run_from_config()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
