def __getattr__(name):
    if name in {"SampleDataPack", "build_id_datapack", "load_zen_data"}:
        from .datapack import SampleDataPack, build_id_datapack, load_zen_data

        return {
            "SampleDataPack": SampleDataPack,
            "build_id_datapack": build_id_datapack,
            "load_zen_data": load_zen_data,
        }[name]

    if name == "load_hh_h5_list":
        from .rawdataloader.hh_data import load_hh_h5_list

        return load_hh_h5_list

    if name in {"ensure_dir", "model_save"}:
        from .io import ensure_dir, model_save

        return {"ensure_dir": ensure_dir, "model_save": model_save}[name]

    if name in {
        "drop_tensor_names",
        "restore_td_names_like",
        "restore_tensor_names_like",
        "strip_td_names",
        "strip_tensor_names",
    }:
        from .named_tensors import (
            drop_tensor_names,
            restore_td_names_like,
            restore_tensor_names_like,
            strip_td_names,
            strip_tensor_names,
        )

        return {
            "drop_tensor_names": drop_tensor_names,
            "restore_td_names_like": restore_td_names_like,
            "restore_tensor_names_like": restore_tensor_names_like,
            "strip_td_names": strip_td_names,
            "strip_tensor_names": strip_tensor_names,
        }[name]

    if name == "SliceableTensorDict":
        from .sliceable_tensordict import SliceableTensorDict

        return SliceableTensorDict

    if name in {"_collate", "stack_name", "td_cat"}:
        from .tensordict_ops import _collate, stack_name, td_cat

        return {"_collate": _collate, "stack_name": stack_name, "td_cat": td_cat}[name]

    if name == "fingerprint":
        from .utils import fingerprint

        return fingerprint

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "SliceableTensorDict",
    "SampleDataPack",
    "load_zen_data",
    "load_hh_h5_list",
    "build_id_datapack",
    "model_save",
    "ensure_dir",
    "drop_tensor_names",
    "strip_tensor_names",
    "restore_tensor_names_like",
    "strip_td_names",
    "restore_td_names_like",
    "_collate",
    "td_cat",
    "stack_name",
    "fingerprint",
]
