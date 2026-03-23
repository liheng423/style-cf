from .config_loader import get_platoon_config, get_platoon_configs


def run_platoon_experiments(*args, **kwargs):
    from .runner import run_platoon_experiments as _run

    return _run(*args, **kwargs)


__all__ = ["get_platoon_configs", "get_platoon_config", "run_platoon_experiments"]
