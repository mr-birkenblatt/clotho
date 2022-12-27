# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level
from typing import Any, Dict, List

from _typeshed import Incomplete
from torch.utils.data.datapipes.datapipe import DFIterDataPipe, IterDataPipe


class DataFrameTracedOps(DFIterDataPipe):
    source_datapipe: Incomplete
    output_var: Incomplete
    def __init__(self, source_datapipe, output_var) -> None: ...
    def __iter__(self): ...


class Capture:
    ctx: Dict[str, List[Any]]
    def __init__(self) -> None: ...
    def __getattr__(self, attrname): ...
    def __getitem__(self, key): ...
    def __setitem__(self, key, value) -> None: ...
    def __add__(self, add_val): ...
    def __sub__(self, add_val): ...
    def __mul__(self, add_val): ...


class CaptureF(Capture):
    ctx: Incomplete
    kwargs: Incomplete
    def __init__(self, ctx: Incomplete | None = ..., **kwargs) -> None: ...


class CaptureCall(CaptureF):
    def execute(self): ...


class CaptureVariableAssign(CaptureF):
    def execute(self) -> None: ...


class CaptureVariable(Capture):
    value: Incomplete
    name: Incomplete
    calculated_value: Incomplete
    names_idx: int
    ctx: Incomplete
    def __init__(self, value, ctx) -> None: ...
    def execute(self): ...
    def apply_ops(self, dataframe): ...


class CaptureGetItem(Capture):
    left: Capture
    key: Any
    ctx: Incomplete
    def __init__(self, left, key, ctx) -> None: ...
    def execute(self): ...


class CaptureSetItem(Capture):
    left: Capture
    key: Any
    value: Capture
    ctx: Incomplete
    def __init__(self, left, key, value, ctx) -> None: ...
    def execute(self) -> None: ...


class CaptureAdd(Capture):
    left: Incomplete
    right: Incomplete
    ctx: Incomplete
    def __init__(self, left, right, ctx) -> None: ...
    def execute(self): ...


class CaptureMul(Capture):
    left: Incomplete
    right: Incomplete
    ctx: Incomplete
    def __init__(self, left, right, ctx) -> None: ...
    def execute(self): ...


class CaptureSub(Capture):
    left: Incomplete
    right: Incomplete
    ctx: Incomplete
    def __init__(self, left, right, ctx) -> None: ...
    def execute(self): ...


class CaptureGetAttr(Capture):
    source: Incomplete
    name: str
    ctx: Incomplete
    src: Incomplete
    def __init__(self, src, name, ctx) -> None: ...
    def execute(self): ...


def get_val(capture): ...


class CaptureInitial(CaptureVariable):
    name: Incomplete
    def __init__(self) -> None: ...


class CaptureDataFrame(CaptureInitial):
    ...


class CaptureDataFrameWithDataPipeOps(CaptureDataFrame):
    def as_datapipe(self): ...
    def raw_iterator(self): ...
    def __iter__(self): ...

    def batch(
        self,
        batch_size: int = ..., drop_last: bool = ..., wrapper_class=...): ...

    def groupby(
        self, group_key_fn, *, buffer_size: int = ...,
        group_size: Incomplete | None = ...,
        guaranteed_group_size: Incomplete | None = ...,
        drop_remaining: bool = ...): ...

    def shuffle(self, *args, **kwargs): ...
    def filter(self, *args, **kwargs): ...
    def __getattr__(self, attrname): ...


class DataFrameTracer(CaptureDataFrameWithDataPipeOps, IterDataPipe):
    source_datapipe: Incomplete
    def __init__(self, source_datapipe) -> None: ...
