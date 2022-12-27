# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


from typing import Any, Callable, Dict, Iterable, List, Set

from _typeshed import Incomplete
from torch.utils._mode_utils import _ModeInfo, MetaInitErrorInfo


def get_ignored_functions() -> Set[Callable]: ...


def get_testing_overrides() -> Dict[Callable, Callable]: ...


def wrap_torch_function(dispatcher: Callable): ...


def handle_torch_function(
    public_api: Callable, relevant_args: Iterable[Any], *args,
    **kwargs) -> Any: ...


has_torch_function: Incomplete


def get_overridable_functions() -> Dict[Any, List[Callable]]: ...


def resolve_name(f): ...


def is_tensor_method_or_property(func: Callable) -> bool: ...


def is_tensor_like(inp): ...


class _TorchFunctionMetaInitErrorInfo(MetaInitErrorInfo):
    def __init__(self) -> None: ...


class TorchFunctionModeMeta(type):
    def __new__(metacls, name, bases, dct): ...


class TorchFunctionMode(metaclass=TorchFunctionModeMeta):
    inner: TorchFunctionMode
    def __init__(self) -> None: ...

    def __torch_function__(
        self, func, types, args=...,
        kwargs: Incomplete | None = ...) -> None: ...

    @classmethod
    def push(cls, *args, **kwargs): ...


class BaseTorchFunctionMode(TorchFunctionMode):

    def __torch_function__(
        self, func, types, args=..., kwargs: Incomplete | None = ...): ...


class _TorchFunctionModeInfo(_ModeInfo):
    def __init__(self) -> None: ...
    def get_mode(self): ...
    def set_mode(self, mode): ...


class enable_reentrant_dispatch:
    def __enter__(self) -> None: ...

    def __exit__(
        self, exc_type: Any, exc_value: Any, traceback: Any) -> None: ...
