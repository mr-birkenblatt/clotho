# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


from typing import (
    Any,
    Callable,
    Dict,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

import torch
from _typeshed import Incomplete
from torch._C import ListType as ListType
from torch._C import TupleType as TupleType
from torch.jit._recursive import wrap_cpp_module as wrap_cpp_module


T = TypeVar('T')
MAX_RAW_TENSOR_SIZE: int


class InflatableArg(NamedTuple):
    value: Any
    fmt: str
    fmt_fn: str


def bundle_inputs(
    model: torch.jit.ScriptModule, inputs: Union[Optional[Sequence[Tuple[Any,
                                    ...]]], Dict[Callable,
                    Optional[Sequence[Tuple[Any, ...]]]]],
    info: Optional[Union[List[str], Dict[Callable, List[str]]]] = ..., *,
    _receive_inflate_expr: Optional[List[
                    str]] = ...) -> torch.jit.ScriptModule: ...


def augment_model_with_bundled_inputs(
    model: torch.jit.ScriptModule, inputs: Optional[Sequence[Tuple[Any,
                            ...]]] = ...,
    _receive_inflate_expr: Optional[List[str]] = ...,
    info: Optional[List[str]] = ..., skip_size_check: bool = ...) -> None: ...


def augment_many_model_functions_with_bundled_inputs(
    model: torch.jit.ScriptModule, inputs: Dict[Callable,
            Optional[Sequence[Tuple[Any, ...]]]],
    _receive_inflate_expr: Optional[List[str]] = ...,
    info: Optional[Dict[Callable, List[str]]] = ...,
    skip_size_check: bool = ...) -> None: ...


def bundle_randn(*size, dtype: Incomplete | None = ...): ...


def bundle_large_tensor(t): ...
