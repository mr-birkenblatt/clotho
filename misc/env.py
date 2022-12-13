import os
from typing import Literal


EnvPath = Literal[
    "USER_PATH",
]
EnvStr = Literal[
    "API_REDIS_HOST",
    "API_REDIS_PASSWD",
    "API_REDIS_PREFIX",
    "API_SERVER_HOST",
    "API_SERVER_NAMESPACE",
]
EnvInt = Literal[
    "API_REDIS_PORT",
    "API_SERVER_PORT",
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
