import os
from typing import Literal


EnvPath = Literal[
    "MSG_PATH",
    "MSG_TOPICS",
    "USER_PATH",
]
EnvStr = Literal[
    "HOST",
    "LINK_STORE",
    "MSG_STORE",
    "REDIS_HOST",
    "REDIS_PASS",
    "USER_STORE",
]
EnvInt = Literal[
    "PORT",
    "REDIS_PORT",
]


def _envload(key: str, default: str | None) -> str:
    res = os.environ.get(key)
    if res is not None:
        return res
    if default is not None:
        return default
    raise ValueError(f"env {key} must be set!")


def envload_str(key: EnvStr, *, default: str | None = None) -> str:
    return _envload(key, default)


def envload_path(key: EnvPath, *, default: str | None = None) -> str:
    return _envload(key, default)


def envload_int(key: EnvInt, *, default: int | None = None) -> int:
    return int(_envload(key, f"{default}"))
