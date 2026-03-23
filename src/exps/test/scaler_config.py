from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from ...schema import FEATNAMES as FEAT
from ..utils.scaler_io import load_scaler_payload


def _resolve_scaler_path(test_config: Mapping[str, Any], config_key: str, default_rel_path: str) -> Path:
    raw = test_config.get(config_key, default_rel_path)
    if raw in (None, "", ...):
        raise ValueError(
            f"Missing scaler path in test_config['{config_key}']. "
            "Please set it in the active testing config."
        )

    path = Path(str(raw))
    if not path.exists():
        raise FileNotFoundError(f"Scaler file not found for {config_key}: {path}")
    return path


def _coerce_scaler_mapping(payload: Any, expected_keys: Sequence[str], source: Path) -> dict[str, object]:
    if isinstance(payload, Mapping):
        missing = [key for key in expected_keys if key not in payload]
        if missing:
            raise ValueError(
                f"Scaler file {source} is missing keys {missing}. "
                f"Expected keys: {list(expected_keys)}."
            )
        return {key: payload[key] for key in expected_keys}

    if isinstance(payload, (list, tuple)):
        if len(payload) < len(expected_keys):
            raise ValueError(
                f"Scaler file {source} has {len(payload)} items but needs at least {len(expected_keys)} "
                f"for keys {list(expected_keys)}."
            )
        return {key: payload[idx] for idx, key in enumerate(expected_keys)}

    if len(expected_keys) == 1:
        return {expected_keys[0]: payload}

    raise TypeError(
        f"Unsupported scaler payload type from {source}: {type(payload).__name__}. "
        "Expected mapping or sequence."
    )


def load_test_scalers(
    test_config: Mapping[str, Any],
    style_x_groups: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    style_path = _resolve_scaler_path(test_config, "style_scaler_path", "data/saved_scalers/scaler_stylecf.pkl")
    transformer_path = _resolve_scaler_path(
        test_config,
        "transformer_scaler_path",
        "data/saved_scalers/scaler_transfollower.pkl",
    )
    lstm_path = _resolve_scaler_path(test_config, "lstm_scaler_path", "data/saved_scalers/scaler_lstm.pkl")

    style_payload = load_scaler_payload(style_path)
    transformer_payload = load_scaler_payload(transformer_path)
    lstm_payload = load_scaler_payload(lstm_path)

    style_keys = tuple(style_x_groups.keys())
    transformer_keys = ("enc_x", "dec_x")
    lstm_keys = (FEAT.INPUTS,)

    style_scalers = _coerce_scaler_mapping(style_payload, style_keys, style_path)
    transformer_scalers = _coerce_scaler_mapping(transformer_payload, transformer_keys, transformer_path)
    lstm_scalers = _coerce_scaler_mapping(lstm_payload, lstm_keys, lstm_path)

    return style_scalers, transformer_scalers, lstm_scalers


__all__ = ["load_test_scalers"]
