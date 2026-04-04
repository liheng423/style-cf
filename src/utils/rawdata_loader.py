from __future__ import annotations

from pathlib import Path
from typing import Mapping

try:
    import tomllib  # type: ignore[attr-defined]
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib  # type: ignore[no-redef]


RAWDATA_CONFIG_DIR = Path(__file__).resolve().parents[1] / "exps" / "configs" / "rawdata_cfgs"

class _FixedRoller:
    def __init__(self, window_len: int, jump_len: int):
        self.window_len = int(window_len)
        self._jump_len = int(jump_len)

    def jump(self) -> int:
        return self._jump_len


def _normalize_rawdata_config_name(name: str) -> str:
    if not name.lower().endswith(".toml"):
        name = f"{name}.toml"
    return Path(name).name


def load_rawdata_config(rawdata_config: str) -> dict[str, object]:
    path = RAWDATA_CONFIG_DIR / _normalize_rawdata_config_name(rawdata_config)
    payload = tomllib.loads(path.read_text(encoding="utf-8-sig"))
    raw_cfg = payload.get("rawdata")
    if not isinstance(raw_cfg, Mapping):
        raise KeyError(f"Missing [rawdata] table in rawdata config: {path}")
    return dict(raw_cfg)


def _source_data_filename(raw_cfg: Mapping[str, object]) -> str:
    extractor = str(raw_cfg["extractor"]).lower()
    if extractor == "zen":
        return Path(str(raw_cfg["datapath"])).name
    if extractor == "hh_h5_list":
        raw_paths = raw_cfg["datapath"]
        if not isinstance(raw_paths, list):
            raise TypeError("hh_h5_list rawdata.datapath must be a list of paths.")
        return "|".join(Path(str(p)).name for p in raw_paths)
    raise ValueError(f"Unsupported rawdata extractor: {extractor}")


def load_datapack(rawdata_config: str):
    raw_cfg = load_rawdata_config(rawdata_config)
    extractor = str(raw_cfg["extractor"]).lower()
    rise = bool(raw_cfg["rise"])
    in_kph = bool(raw_cfg["in_kph"])
    kilo_norm = bool(raw_cfg["kilo_norm"])

    if extractor == "hh_h5_list":
        from ..exps.utils import load_hh_h5_list

        paths = [str(p) for p in raw_cfg["datapath"]]
        window_len = int(raw_cfg["hh_window_len"])
        jump_len = int(raw_cfg["hh_window_jump"])
        roller = _FixedRoller(window_len=window_len, jump_len=jump_len)
        key = raw_cfg.get("h5_key")
        datapack = load_hh_h5_list(
            paths=paths,
            rise=rise,
            roller=roller,
            key=None if key in (None, "") else str(key),
            in_kph=in_kph,
            kilo_norm=kilo_norm,
            dt=float(raw_cfg["hh_dt"]),
        )
        return datapack, dict(raw_cfg), _source_data_filename(raw_cfg)

    if extractor == "zen":
        from ..exps.utils import load_zen_data

        datapack = load_zen_data(str(raw_cfg["datapath"]), rise=rise, in_kph=in_kph, kilo_norm=kilo_norm)
        return datapack, dict(raw_cfg), _source_data_filename(raw_cfg)

    raise ValueError(f"Unsupported rawdata extractor: {extractor}")


__all__ = [
    "load_rawdata_config",
    "load_datapack",
]
