def get_style_configs(*args, **kwargs):
    from ..utils.config_utils import get_style_configs as _get_style_configs

    return _get_style_configs(*args, **kwargs)


def load_common_config(*args, **kwargs):
    from ..utils.config_utils import load_common_config as _load_common_config

    return _load_common_config(*args, **kwargs)


def load_style_pipeline_config(*args, **kwargs):
    from ..utils.config_utils import load_style_pipeline_config as _load_style_pipeline_config

    return _load_style_pipeline_config(*args, **kwargs)


def resolve_common_runtime(*args, **kwargs):
    from ..utils.config_utils import resolve_common_runtime as _resolve_common_runtime

    return _resolve_common_runtime(*args, **kwargs)


__all__ = ["get_style_configs", "load_common_config", "load_style_pipeline_config", "resolve_common_runtime"]
