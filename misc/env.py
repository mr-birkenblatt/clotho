import os
from typing import Literal, Optional


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


def _envload(key: str, default: Optional[str]) -> str:
    res = os.environ.get(key)
    if res is not None:
        return res
    if default is not None:
        return default
    raise ValueError(f"env {key} must be set!")


def envload_str(key: EnvStr, *, default: Optional[str] = None) -> str:
    return _envload(key, default)


def envload_path(key: EnvPath, *, default: Optional[str] = None) -> str:
    return _envload(key, default)


def envload_int(key: EnvInt, *, default: Optional[int] = None) -> int:
    return int(_envload(key, f"{default}"))
