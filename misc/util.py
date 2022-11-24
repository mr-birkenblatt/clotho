import hashlib
import inspect
import json
import os
import string
from typing import Any, IO, Iterable, Type, TypeVar

import numpy as np
import pandas as pd


CT = TypeVar('CT')
RT = TypeVar('RT')
VT = TypeVar('VT')
DT = TypeVar('DT', bound=pd.DataFrame | pd.Series)


def get_text_hash(text: str) -> str:
    blake = hashlib.blake2b(digest_size=32)
    blake.update(text.encode("utf-8"))
    return blake.hexdigest()


def get_short_hash(text: str) -> str:
    blake = hashlib.blake2b(digest_size=4)
    blake.update(text.encode("utf-8"))
    return blake.hexdigest()


def is_hex(text: str) -> bool:
    hex_digits = set(string.hexdigits)
    return all(char in hex_digits for char in text)


def as_df(series: pd.Series) -> pd.DataFrame:
    return series.to_frame().T


def fillnonnum(df: DT, val: float) -> DT:
    return df.replace([-np.inf, np.inf], np.nan).fillna(val)


def only(arr: list[RT]) -> RT:
    if len(arr) != 1:
        raise ValueError(f"array must have exactly one element: {arr}")
    return arr[0]


# time units for logging request durations
ELAPSED_UNITS: list[tuple[int, str]] = [
    (1, "s"),
    (60, "m"),
    (60*60, "h"),
    (60*60*24, "d"),
]


def elapsed_time_string(elapsed: float) -> str:
    """Convert elapsed time into a readable string."""
    cur = ""
    for (conv, unit) in ELAPSED_UNITS:
        if elapsed / conv >= 1 or not cur:
            cur = f"{elapsed / conv:8.3f}{unit}"
        else:
            break
    return cur


def to_bool(value: bool | float | int | str) -> bool:
    value = f"{value}".lower()
    if value == "true":
        return True
    if value == "false":
        return False
    try:
        return bool(int(float(value)))
    except ValueError:
        pass
    raise ValueError(f"{value} cannot be interpreted as bool")


def to_list(value: Any) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{value} is not a list")
    return value


def is_int(value: Any) -> bool:
    try:
        int(value)
        return True
    except ValueError:
        return False


def is_float(value: Any) -> bool:
    try:
        float(value)
        return True
    except (ValueError, TypeError):
        return False


def is_json(value: str) -> bool:
    try:
        json.loads(value)
    except json.JSONDecodeError:
        return False
    return True


def report_json_error(err: json.JSONDecodeError) -> None:
    raise ValueError(
        f"JSON parse error ({err.lineno}:{err.colno}): "
        f"{repr(err.doc)}") from err


def json_maybe_read(data: str) -> Any | None:
    try:
        return json.loads(data)
    except json.JSONDecodeError:
        return None


def json_load(fin: IO[str]) -> Any:
    try:
        return json.load(fin)
    except json.JSONDecodeError as e:
        report_json_error(e)
        raise e


def json_dump(obj: Any, fout: IO[str]) -> None:
    print(json_pretty(obj), file=fout)


def json_pretty(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, indent=2)


def json_compact(obj: Any) -> bytes:
    return json.dumps(
        obj,
        sort_keys=True,
        indent=None,
        separators=(',', ':')).encode("utf-8")


def json_read(data: bytes) -> Any:
    try:
        return json.loads(data.decode("utf-8"))
    except json.JSONDecodeError as e:
        report_json_error(e)
        raise e


def read_jsonl(fin: IO[str]) -> Iterable[Any]:
    for line in fin:
        line = line.rstrip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError as e:
            report_json_error(e)
            raise e


UNIX_EPOCH = pd.Timestamp("1970-01-01", tz="UTC")


def from_timestamp(timestamp: float) -> pd.Timestamp:
    return pd.to_datetime(timestamp, unit="s", utc=True)


def to_timestamp(time: pd.Timestamp) -> float:
    return (time - UNIX_EPOCH) / pd.Timedelta("1s")


def now_ts() -> pd.Timestamp:
    return pd.Timestamp("now", tz="UTC")


def get_function_info(*, clazz: Type) -> tuple[str, int, str]:
    stack = inspect.stack()

    def get_method(cur_clazz: Type) -> tuple[str, int, str] | None:
        class_filename = inspect.getfile(cur_clazz)
        for level in stack:
            if os.path.samefile(level.filename, class_filename):
                return level.filename, level.lineno, level.function
        return None

    queue = [clazz]
    while queue:
        cur = queue.pop(0)
        res = get_method(cur)
        if res is not None:
            return res
        queue.extend(cur.__bases__)
    frame = stack[1]
    return frame.filename, frame.lineno, frame.function


def get_relative_function_info(
        depth: int) -> tuple[str, int, str, dict[str, Any]]:
    depth += 1
    stack = inspect.stack()
    if depth >= len(stack):
        return "unknown", -1, "unknown", {}
    frame = stack[depth]
    return frame.filename, frame.lineno, frame.function, frame.frame.f_locals
