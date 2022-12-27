# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access,no-name-in-module,undefined-variable


from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch
import torch.fx as fx
from _typeshed import Incomplete
from torch.fx import GraphModule, Tracer
from torch.utils._python_dispatch import TorchDispatchMode


class ProxyTensor(torch.Tensor):
    proxy: fx.Proxy
    @staticmethod
    def __new__(cls, elem, proxy): ...
    __torch_function__: Incomplete

    @classmethod
    def __torch_dispatch__(
        cls, func_overload, types, args=...,
        kwargs: Incomplete | None = ...): ...


class PythonKeyTracer(Tracer):
    def __init__(self) -> None: ...

    def call_module(
        self, m: torch.nn.Module, forward: Callable[..., Any],
        args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any: ...

    def create_arg(self, a: Any): ...


def dispatch_trace(
    root: Union[torch.nn.Module, Callable],
    concrete_args: Optional[Tuple[Any, ...]] = ...,
    trace_factory_functions: bool = ...) -> GraphModule: ...


class ProxyTorchDispatchMode(TorchDispatchMode):
    tracer: Incomplete
    def __init__(self, tracer) -> None: ...

    def __torch_dispatch__(
        self, func_overload, types, args=...,
        kwargs: Incomplete | None = ...): ...


def make_fx(
    f, decomposition_table: Incomplete | None = ...,
    trace_factory_functions: bool = ...): ...
