# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


from typing import Any, NamedTuple

from _typeshed import Incomplete

from .grad_mode import _DecoratorContextManager


def enter_dual_level(): ...


def exit_dual_level(*, level: Incomplete | None = ...) -> None: ...


def make_dual(tensor, tangent, *, level: Incomplete | None = ...): ...


class _UnpackedDualTensor(NamedTuple):
    primal: Incomplete
    tangent: Incomplete


class UnpackedDualTensor(_UnpackedDualTensor):
    ...


def unpack_dual(tensor, *, level: Incomplete | None = ...): ...


class dual_level(_DecoratorContextManager):
    def __init__(self) -> None: ...
    def __enter__(self): ...

    def __exit__(
        self, exc_type: Any, exc_value: Any, traceback: Any) -> None: ...
