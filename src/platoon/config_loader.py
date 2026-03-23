from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

try:
    import tomllib  # type: ignore[attr-defined]
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]


_ROOT = Path(__file__).resolve().parent
_CONFIG_DIR = _ROOT / "config"
_DEFAULT_DIR = _CONFIG_DIR / "default_configs"
_DEFAULT_FILES = (
    "simulation.toml",
    "newell.toml",
    "evaluation.toml",
    "experiments.toml",
)

_CACHE: dict[str, Any] | None = None


def _load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        return tomllib.load(f)


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = dict(base)
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            merged[key] = _deep_merge(current, value)
        else:
            merged[key] = value
    return merged


def _load_defaults() -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for filename in _DEFAULT_FILES:
        path = _DEFAULT_DIR / filename
        if not path.exists():
            continue
        payload = _deep_merge(payload, _load_toml(path))
    return payload


def _load_active_override() -> dict[str, Any]:
    active_path = _CONFIG_DIR / "active.toml"
    if not active_path.exists():
        return {}

    active_payload = _load_toml(active_path)
    section = active_payload.get("platoon")
    if not isinstance(section, Mapping):
        return {}

    active_name = section.get("active")
    if not isinstance(active_name, str) or not active_name.strip():
        return {}

    filename = active_name.strip()
    if not filename.lower().endswith(".toml"):
        filename = f"{filename}.toml"
    filename = Path(filename).name

    override_path = _CONFIG_DIR / filename
    if not override_path.exists():
        return {}
    return _load_toml(override_path)


def _build_bundle() -> dict[str, Any]:
    defaults = _load_defaults()
    override = _load_active_override()
    merged = _deep_merge(defaults, override)

    for section_name in ("simulation_config", "newell_config", "evaluation_config", "experiments"):
        if section_name not in merged:
            merged[section_name] = {}
    return merged


def get_platoon_configs(force_reload: bool = False) -> dict[str, Any]:
    global _CACHE
    if _CACHE is None or force_reload:
        _CACHE = _build_bundle()
    return _CACHE


def get_platoon_config(name: str, force_reload: bool = False) -> Any:
    bundle = get_platoon_configs(force_reload=force_reload)
    if name not in bundle:
        raise KeyError(f"Unknown platoon config section: {name}")
    return bundle[name]


__all__ = ["get_platoon_configs", "get_platoon_config"]
