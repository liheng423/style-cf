from __future__ import annotations

import argparse
from typing import Mapping

from .exps.datahandle.data_spliter import build_split_indices
from .exps.datahandle.datapackbuilder import build_dataset
from .utils.config_utils import get_datahandle_config, get_datasplit_configs
from .utils.logger import logger
from .utils.rawdata_loader import load_datapack


def _load_filtered_datapack(split_cfg: Mapping[str, object]):
    datapack, raw_cfg, source_name = load_datapack(str(split_cfg["rawdata_config"]))
    datahandle_cfg = get_datahandle_config()
    filtered = build_dataset(
        datapack,
        list(datahandle_cfg["filter_names"]),
        dict(datahandle_cfg["data_filter_config"]),
    )

    logger.info(
        "Data split source prepared | "
        f"extractor={raw_cfg.get('extractor')} source={source_name} "
        f"filtered_samples={int(filtered.data.shape[0])}"
    )
    return filtered


def run_data_split(split_cfg: Mapping[str, object] | None = None):
    cfg = dict(split_cfg) if split_cfg is not None else dict(get_datasplit_configs()["data_split_config"])
    if "split_index_path" not in cfg and "split_base_path" in cfg:
        cfg["split_index_path"] = cfg["split_base_path"]
    datapack = _load_filtered_datapack(cfg)
    seed = int(cfg.get("split_seed", 42))

    train_idx, val_idx, test_idx = build_split_indices(
        d=datapack,
        data_config=cfg,
        seed=seed,
        save=True,
    )

    logger.info(
        "Data split done | "
        f"train={len(train_idx)} val={len(val_idx)} test={len(test_idx)} "
        f"save_path={cfg.get('split_base_path', cfg.get('split_index_path'))}"
    )
    return train_idx, val_idx, test_idx


def _build_cli_config(args: argparse.Namespace) -> dict[str, object]:
    cfg = dict(get_datasplit_configs()["data_split_config"])

    if args.rawdata_config is not None:
        cfg["rawdata_config"] = args.rawdata_config
    if args.output is not None:
        cfg["split_base_path"] = args.output
        cfg["split_index_path"] = args.output
    if args.split_mode is not None:
        cfg["split_mode"] = args.split_mode
    if args.seed is not None:
        cfg["split_seed"] = args.seed
    if args.train_ratio is not None:
        cfg["train_ratio"] = args.train_ratio
    if args.val_ratio is not None:
        cfg["val_ratio"] = args.val_ratio
    if args.test_ratio is not None:
        cfg["test_ratio"] = args.test_ratio

    if "split_index_path" not in cfg and "split_base_path" in cfg:
        cfg["split_index_path"] = cfg["split_base_path"]
    return cfg


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build and persist train/val/test split indices.",
    )
    parser.add_argument(
        "--rawdata-config",
        help="Rawdata profile name or TOML filename. Falls back to split_data.toml.",
    )
    parser.add_argument(
        "--output",
        help="Output path for saved split indices. Overrides split_base_path/split_index_path.",
    )
    parser.add_argument(
        "--split-mode",
        choices=("random", "group_self_id"),
        help="Split strategy. Falls back to split_data.toml.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for the split.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        help="Train split ratio.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        help="Validation split ratio.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        help="Test split ratio.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    run_data_split(_build_cli_config(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
