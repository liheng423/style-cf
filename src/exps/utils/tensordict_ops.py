from __future__ import annotations

from typing import Iterable, List, Sequence, cast

import torch
from tensordict import TensorDict
from torch.utils.data._utils.collate import default_collate

from ...stylecf.schema import TensorNames


def _stack_named_tensors(tensors):
    names = tensors[0].names
    unnamed = [t.rename(None) if t.names is not None else t for t in tensors]
    stacked = torch.stack(unnamed, dim=0)
    if names is not None:
        stacked = stacked.refine_names(TensorNames.N, *names)
    return stacked


def _stack_tensordict(batch):
    if not batch:
        return TensorDict({}, batch_size=[0])
    first = batch[0]
    stacked = {}
    for key in first.keys():
        tensors = [td[key] for td in batch]
        stacked[key] = _stack_named_tensors(tensors)
    return TensorDict(stacked, batch_size=[len(batch)])


def _collate(batch):
    if not batch:
        return batch
    first = batch[0]
    if isinstance(first, TensorDict):
        return _stack_tensordict(batch)
    if (
        isinstance(first, tuple)
        and len(first) == 2
        and isinstance(first[0], TensorDict)
        and isinstance(first[1], TensorDict)
    ):
        xs, ys = zip(*batch)
        return _stack_tensordict(list(xs)), _stack_tensordict(list(ys))
    return default_collate(batch)


def td_cat(tensordicts: list[TensorDict], dim) -> TensorDict:
    td = cast(list, tensordicts)
    return cast(TensorDict, torch.cat(td, dim=dim))


def stack_name(tensordict_list: Sequence[TensorDict], dim_name: str):
    """
    Stacks a TensorDict along the specified named dimension.
    """
    first_type = type(tensordict_list[0])
    if not all(isinstance(td, first_type) for td in tensordict_list):
        raise TypeError("All items in tensordict_list must have the same class")

    if not tensordict_list:
        return TensorDict({}, batch_size=[])

    keys = list(cast(Iterable[str], tensordict_list[0].keys()))
    assert any(
        tensordict_list[0][key].names is not None and dim_name in tensordict_list[0][key].names
        for key in keys
    ), f"Dimension name '{dim_name}' not found in the names of any tensor in the first TensorDict."

    stacked_data = {}
    for key in cast(Iterable[str], keys):
        tensors_to_stack = []
        for td in tensordict_list:
            if key in td.keys():
                tensors_to_stack.append(td[key])
            else:
                raise ValueError(f"Key '{key}' not found in all TensorDicts.")

        if dim_name in tensordict_list[0][key].names:
            stacked_data[key] = torch.concat(tensors_to_stack, dim=dim_name)
        else:
            stacked_data[key] = tensors_to_stack[0]

    new_batch_size = list(tensordict_list[0].batch_size)
    names = cast(List[str], tensordict_list[0].names)
    if dim_name in names:
        dim_idx = names.index(dim_name)
        new_batch_size[dim_idx] *= len(tensordict_list)

    return first_type(stacked_data, batch_size=new_batch_size, names=names)


__all__ = ["_collate", "td_cat", "stack_name"]
