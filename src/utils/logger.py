
import time
from functools import wraps
from src.exps.configs import style_train_config

class Logger:
    MODE_SILENT = "silent"
    MODE_NORMAL = "normal"

    LEVELS = {
        "debug": 10,
        "info": 20,
        "warn": 30,
        "error": 40,
    }

    def __init__(self, mode=MODE_NORMAL):
        self.set_mode(mode)

    def set_mode(self, mode):
        if mode not in (self.MODE_SILENT, self.MODE_NORMAL):
            raise ValueError(f"Unknown mode: {mode}")
        self.mode = mode
        # Map mode to minimum level to print.
        if mode == self.MODE_SILENT:
            self.min_level = None
        else:
            self.min_level = self.LEVELS["info"]

    def _should_print(self, level):
        if self.min_level is None:
            return False
        return level >= self.min_level

    def print(self, *args, **kwargs):
        if self._should_print(self.LEVELS["info"]):
            print(*args, **kwargs)

    def info(self, *args, **kwargs):
        self.print(*args, **kwargs)

    def debug(self, *args, **kwargs):
        if self._should_print(self.LEVELS["debug"]):
            print(*args, **kwargs)

    def warn(self, *args, **kwargs):
        if self._should_print(self.LEVELS["warn"]):
            print(*args, **kwargs)

    def error(self, *args, **kwargs):
        if self._should_print(self.LEVELS["error"]):
            print(*args, **kwargs)

    def decorator(self, name=None):
        def _decorator(func):
            label = name or func.__name__

            @wraps(func)
            def _wrapper(*args, **kwargs):
                start = time.time()
                self.print(f"[{label}] start")
                try:
                    return func(*args, **kwargs)
                finally:
                    elapsed = time.time() - start
                    self.print(f"[{label}] end ({elapsed:.2f}s)")

            return _wrapper

        return _decorator


logger = Logger(style_train_config["logger"]["mode"])
