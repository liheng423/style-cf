from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, cast

import matplotlib.pyplot as plt
import torch

from .exps.configs import (
    DEFAULT_STYLE_WINDOW,
    DEFAULT_TEST_WINDOW,
    data_filter_config,
    filter_names,
    lstm_data_config,
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
from .exps.test.visualizer import plot_error_evolution, plot_metric_histograms
from .exps.utils.utils import load_zen_data


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
    start_time: int
    style_token_seconds: float
    style_token_mode: str
    output_dir: Path
    enabled_models: tuple[str, ...]
    save_results: bool
    plot_results: bool


@dataclass
class EvalBundle:
    d_style: object
    d_test: object
    style_scalers: dict[str, object]
    transformer_scalers: dict[str, object]
    lstm_scalers: dict[str, object]


def _resolve_device() -> torch.device:
    device = test_config.get("device")
    if isinstance(device, str):
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _resolve_data_path() -> str:
    env_path = os.environ.get("ZEN_DATA_PATH")
    path = env_path or test_config.get("datapath")
    if not path:
        raise ValueError("Dataset path is missing. Set ZEN_DATA_PATH or test_config['datapath']")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")
    return path


def _dataset(head: int | None = None):
    d = load_zen_data(_resolve_data_path(), rise=True, in_kph=False, kilo_norm=True)
    return d.head(head) if head is not None else d


def _build_options() -> TestingOptions:
    head_env = os.environ.get("TEST_HEAD")
    head = int(head_env) if head_env else None

    style_window = tuple(test_config.get("style_window", DEFAULT_STYLE_WINDOW))
    test_window = tuple(test_config.get("test_window", DEFAULT_TEST_WINDOW))

    style_token_seconds = float(test_config.get("style_token_seconds", 6.0))
    style_token_mode = str(test_config.get("style_token_mode", "per_sample"))
    start_time = int(test_config.get("start_time", 60))

    models_cfg = test_config.get("enabled_models")
    if isinstance(models_cfg, (list, tuple)) and models_cfg:
        enabled_models = tuple(str(x).lower() for x in models_cfg)
    else:
        enabled_models = ["stylecf", "transformer", "idm"]
        if "lstm_agent" in test_config:
            enabled_models.append("lstm")
        enabled_models = tuple(enabled_models)

    output_dir = Path(str(test_config.get("output_dir", "models/test_results")))

    plot_env = os.environ.get("TEST_PLOT")
    if plot_env is not None:
        plot_results = plot_env.strip() in {"1", "true", "True"}
    else:
        plot_results = bool(test_config.get("plot_results", False))

    return TestingOptions(
        head=head,
        style_window=cast(tuple[int, int], style_window),
        test_window=cast(tuple[int, int], test_window),
        start_time=start_time,
        style_token_seconds=style_token_seconds,
        style_token_mode=style_token_mode,
        output_dir=output_dir,
        enabled_models=cast(tuple[str, ...], enabled_models),
        save_results=bool(test_config.get("save_results", True)),
        plot_results=plot_results,
    )


def _build_eval_bundle(
    head: int | None = None,
    style_window: tuple[int, int] = DEFAULT_STYLE_WINDOW,
    test_window: tuple[int, int] = DEFAULT_TEST_WINDOW,
) -> EvalBundle:
    raw_data = _dataset(head=head)
    d_full = build_style_datapack(raw_data, filter_names, data_filter_config)
    d_style, d_test = split_eval_windows(d_full, style_window, test_window)
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
    )
    d_style = eval_bundle.d_style
    d_test = eval_bundle.d_test

    num_samples = d_test.data.shape[0]
    build_traj = build_test_traj_builder(d_test, device)

    results: dict[str, ModelEvalResult] = {}
    context = BuilderContext(
        device=device,
        d_style=d_style,
        d_test=d_test,
        style_scalers=eval_bundle.style_scalers,
        transformer_scalers=eval_bundle.transformer_scalers,
        lstm_scalers=eval_bundle.lstm_scalers,
        style_token_seconds=options.style_token_seconds,
        style_token_mode=options.style_token_mode,
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
