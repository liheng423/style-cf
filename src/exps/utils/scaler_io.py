from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any


def _load_via_joblib(path: Path) -> Any:
    import joblib

    return joblib.load(path)


def _load_via_joblib_alignment_compat(path: Path) -> Any:
    from joblib import numpy_pickle as numpy_pickle_mod
    from joblib.numpy_pickle_utils import _read_bytes

    original_read_array = numpy_pickle_mod.NumpyArrayWrapper.read_array

    def _patched_read_array(self, unpickler):  # type: ignore[no-untyped-def]
        if hasattr(self, "numpy_array_alignment_bytes"):
            pad_len = _read_bytes(unpickler.file_handle, 1, "alignment padding length")[0]
            if pad_len:
                _read_bytes(unpickler.file_handle, int(pad_len), "alignment padding")
        return original_read_array(self, unpickler)

    numpy_pickle_mod.NumpyArrayWrapper.read_array = _patched_read_array
    try:
        with path.open("rb") as fh:
            return numpy_pickle_mod.NumpyUnpickler(str(path), fh, mmap_mode=None).load()
    finally:
        numpy_pickle_mod.NumpyArrayWrapper.read_array = original_read_array


def load_scaler_payload(path: str | Path) -> Any:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Scaler file not found: {p}")

    errors: list[str] = []

    for loader in (_load_via_joblib, _load_via_joblib_alignment_compat):
        try:
            return loader(p)
        except Exception as exc:  # pragma: no cover - depends on runtime versions
            errors.append(f"{loader.__name__}: {type(exc).__name__}: {exc}")

    try:
        with p.open("rb") as fh:
            return pickle.load(fh)
    except Exception as exc:
        errors.append(f"pickle.load: {type(exc).__name__}: {exc}")

    raise ValueError(
        "Failed to load scaler payload from "
        f"{p}. Attempts: {' | '.join(errors)}"
    )


def save_scaler_payload(payload: Any, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        import joblib

        joblib.dump(payload, p)
    except Exception:
        with p.open("wb") as fh:
            pickle.dump(payload, fh)
    return p


__all__ = ["load_scaler_payload", "save_scaler_payload"]
