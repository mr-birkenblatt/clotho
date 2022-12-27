# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


from typing import Iterator

from _typeshed import Incomplete
from torch.utils._mode_utils import _ModeInfo
from torch.utils._mode_utils import MetaInitErrorInfo as MetaInitErrorInfo


class TorchDispatchModeInfo(_ModeInfo):
    def __init__(self) -> None: ...
    def get_mode(self): ...
    def set_mode(self, mode): ...


def enable_torch_dispatch_mode(
    mode, *, replace: Incomplete | None = ...,
        ignore_preexisting: bool = ...) -> Iterator[None]: ...


class TorchDispatchMetaInitErrorInfo(MetaInitErrorInfo):
    def __init__(self) -> None: ...


class TorchDispatchModeMeta(type):
    def __new__(metacls, name, bases, dct): ...


class TorchDispatchMode(metaclass=TorchDispatchModeMeta):
    def __init__(self) -> None: ...

    def __torch_dispatch__(
        self, func, types, args=...,
        kwargs: Incomplete | None = ...) -> None: ...

    @classmethod
    def push(cls, *args, **kwargs): ...


class BaseTorchDispatchMode(TorchDispatchMode):

    def __torch_dispatch__(
        self, func, types, args=..., kwargs: Incomplete | None = ...): ...


def push_torch_dispatch_mode(ctor) -> Iterator[object]: ...
