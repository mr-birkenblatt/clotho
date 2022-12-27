# pylint: disable=multiple-statements,unused-argument, invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias, unused-import
# pylint: disable=redefined-builtin,super-init-not-called, arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors, import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member, keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name, c-extension-no-member
# pylint: disable=protected-access,no-name-in-module, undefined-variable


from typing import Optional, Tuple

from _typeshed import Incomplete
from torch import Tensor as Tensor


def is_sparse(A): ...


def get_floating_dtype(A): ...


def matmul(A: Optional[Tensor], B: Tensor) -> Tensor: ...


def conjugate(A): ...


def transpose(A): ...


def transjugate(A): ...


def bform(X: Tensor, A: Optional[Tensor], Y: Tensor) -> Tensor: ...


def qform(A: Optional[Tensor], S: Tensor): ...


def basis(A): ...


def symeig(
    A: Tensor, largest: Optional[bool] = ...) -> Tuple[Tensor, Tensor]: ...


def solve(
    input: Tensor, A: Tensor, *, out: Incomplete | None = ...) -> Tuple[
        Tensor, Tensor]: ...
