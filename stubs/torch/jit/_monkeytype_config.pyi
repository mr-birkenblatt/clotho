# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


import monkeytype
from _typeshed import Incomplete
from monkeytype.db.base import CallTraceStore, CallTraceStoreLogger


        CallTraceThunk as CallTraceThunk
from types import CodeType
from typing import Dict, Iterable, List, Optional

from monkeytype.tracing import CallTrace as CallTrace
from monkeytype.tracing import CodeFilter as CodeFilter


def is_torch_native_class(cls): ...


def get_type(type): ...


def get_optional_of_element_type(types): ...


def get_qualified_name(func): ...


class JitTypeTraceStoreLogger(CallTraceStoreLogger):
    def __init__(self, store: CallTraceStore) -> None: ...
    def log(self, trace: CallTrace) -> None: ...


class JitTypeTraceStore(CallTraceStore):
    trace_records: Incomplete
    def __init__(self) -> None: ...
    def add(self, traces: Iterable[CallTrace]): ...

    def filter(
        self, qualified_name: str, qualname_prefix: Optional[str] = ...,
        limit: int = ...) -> List[CallTraceThunk]: ...

    def analyze(self, qualified_name: str) -> Dict: ...
    def consolidate_types(self, qualified_name: str) -> Dict: ...
    def get_args_types(self, qualified_name: str) -> Dict: ...


class JitTypeTraceConfig(monkeytype.config.Config):
    s: Incomplete
    def __init__(self, s: JitTypeTraceStore) -> None: ...
    def trace_logger(self) -> JitTypeTraceStoreLogger: ...
    def trace_store(self) -> CallTraceStore: ...
    def code_filter(self) -> Optional[CodeFilter]: ...


class JitTypeTraceStoreLogger:
    def __init__(self) -> None: ...


class JitTypeTraceStore:
    trace_records: Incomplete
    def __init__(self) -> None: ...


class JitTypeTraceConfig:
    def __init__(self) -> None: ...


def jit_code_filter(code: CodeType) -> bool: ...
