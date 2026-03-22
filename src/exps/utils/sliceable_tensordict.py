from __future__ import annotations

from typing import Any, Sequence, cast

import numpy as np
import torch
from tensordict import TensorDict, TensorDictBase

try:
    from typing import Self
except ImportError:  # Python < 3.11
    from typing_extensions import Self


class SliceableTensorDict(TensorDict):
    """
    TensorDict wrapper that keeps subclass type through common operations and
    supports named-dimension slicing with :meth:`sel`.
    """

    def __init__(self, source, batch_size=None, names=None):
        super().__init__(source=source, batch_size=batch_size, names=names)

    def __new__(cls, *args, **kwargs) -> Self:
        return cast(Self, super().__new__(cls))

    @staticmethod
    def _wrap_result(value: Any) -> Any:
        if isinstance(value, SliceableTensorDict):
            return value
        if isinstance(value, TensorDictBase):
            return SliceableTensorDict(
                value,
                batch_size=value.batch_size,
                names=value.names,
            )
        return value

    def __getitem__(self, key: Any) -> Any:
        result = TensorDict.__getitem__(self, cast(Any, key))
        return self._wrap_result(result)

    def get(self, key: Any, default: Any = None) -> Any:
        """
        Dict-like get.

        - Existing key: returns value (or wrapped SliceableTensorDict for key subsets).
        - Missing key: returns ``default``.
        """
        try:
            result = TensorDict.get(self, cast(Any, key))
            return self._wrap_result(result)
        except KeyError:
            return default

    def select_keys(self, *keys: str | Sequence[str]) -> "SliceableTensorDict":
        """
        Return a key subset as a new :class:`SliceableTensorDict`.

        Examples:
            td.select_keys("enc_x", "dec_x")
            td.select_keys(["enc_x", "dec_x"])
        """
        if len(keys) == 1 and isinstance(keys[0], Sequence) and not isinstance(keys[0], str):
            key_list = list(cast(Sequence[str], keys[0]))
        else:
            key_list = [cast(str, k) for k in keys]

        if not key_list:
            raise ValueError("select_keys() requires at least one key")

        return cast("SliceableTensorDict", self[key_list])

    def sel(self, item=None, **indexers) -> "SliceableTensorDict":
        """
        Slice tensors along named dimensions.

        Examples:
            td.sel(T=0)                       # int index
            td.sel(T=slice(1, 5))             # slice index
            td.sel(T=[0, 2, 4], X=slice(0, 3))
            td.sel(("T", 0))                  # backwards-compatible
            td.sel({"T": 0, "X": [1, 3]})     # dict form

        If a named dimension does not exist on a tensor, that tensor is left unchanged.
        """
        if item is not None and indexers:
            raise ValueError("Provide either item or keyword indexers, not both")

        if item is None:
            selectors = indexers
        elif (
            isinstance(item, tuple)
            and len(item) == 2
            and isinstance(item[0], str)
        ):
            selectors = {item[0]: item[1]}
        elif isinstance(item, dict):
            selectors = item
        else:
            raise TypeError(
                "sel() only accepts named-dimension selectors; "
                "use get() or another method for key-based indexing."
            )

        if not selectors:
            return self

        for dim_name in selectors.keys():
            if self.names is not None and dim_name in self.names:
                raise ValueError(f"Slicing along batch axis '{dim_name}' is not allowed")

        def _validate_selector(selector):
            if not isinstance(selector, (int, slice, list, tuple, np.ndarray, torch.Tensor)):
                raise TypeError(
                    "Selector must be int, slice, list/tuple, numpy.ndarray or torch.Tensor when indexing by name"
                )

        for selector in selectors.values():
            _validate_selector(selector)

        def _index_tensor(tensor):
            result = tensor
            for dim_name, selector in selectors.items():
                if result.names is None or dim_name not in result.names:
                    continue
                dim = result.names.index(dim_name)
                if isinstance(selector, int):
                    result = result.select(dim=dim, index=selector)
                    continue

                idx = [slice(None)] * result.ndim
                idx[dim] = (
                    torch.as_tensor(selector, device=result.device)
                    if isinstance(selector, (list, tuple, np.ndarray))
                    else selector
                )
                result = result[tuple(idx)]
            return result

        new_data = {k: _index_tensor(v) for k, v in self.items()}

        return SliceableTensorDict(new_data, batch_size=self.batch_size, names=self.names)

    def isel(self, **indexers) -> "SliceableTensorDict":
        """
        Alias of :meth:`sel` for tensor-style naming.
        """
        return self.sel(**indexers)

    def clone(self, *args, **kwargs) -> "SliceableTensorDict":
        cloned = super().clone(*args, **kwargs)
        return cast("SliceableTensorDict", self._wrap_result(cloned))

    def detach(self) -> "SliceableTensorDict":
        new_data = {k: v.detach() for k, v in self.items()}
        new_td = SliceableTensorDict(new_data, batch_size=self.batch_size, names=self.names)
        return cast("SliceableTensorDict", new_td)

    def to(self, *args, **kwargs):
        new_data = {k: v.to(*args, **kwargs) for k, v in self.items()}
        new_td = SliceableTensorDict(new_data, batch_size=self.batch_size, names=self.names)
        return cast("SliceableTensorDict", new_td)

    def cpu(self) -> "SliceableTensorDict":
        return self.to("cpu")

    def cuda(self, device: int | str | torch.device | None = None) -> "SliceableTensorDict":
        return self.to("cuda" if device is None else device)
