# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


from typing import Any, Callable, Dict, Optional, Union

import torch.fx
from _typeshed import Incomplete


def embedding_override(self, input): ...


def nn_layernorm_override(self, input): ...


def torch_relu_override(x): ...


def torch_nn_relu_override(self, x): ...


def functional_relu_override(x, inplace: bool = ...): ...


def torch_where_override(condition, x, y): ...


def torch_abs_override(input, *, out: Incomplete | None = ...): ...


manual_meta_overrides: Dict[Callable, Callable]


def gen_constructor_wrapper(target): ...


class MetaProxy(torch.fx.Proxy):
    def install_tensor_meta(self, tensor_meta) -> None: ...
    def size(self, dim: Incomplete | None = ...): ...
    def dim(self): ...
    @property
    def shape(self): ...
    @property
    def dtype(self): ...
    @property
    def device(self): ...
    def __getattr__(self, k): ...


class MetaAttribute(MetaProxy):
    root: Incomplete
    attr: Incomplete
    tracer: Incomplete
    def __init__(self, root, attr: str) -> None: ...
    @property
    def node(self): ...
    def __call__(self, *args, **kwargs): ...


class MetaDeviceAttribute(MetaAttribute):
    ...


def proxys_to_metas(v): ...


class MetaTracer(torch.fx.Tracer):
    allow_insert_stateless_mods: bool

    def create_proxy(
        self, kind, target, args, kwargs, name: Incomplete | None = ...,
        type_expr: Incomplete | None = ...,
        proxy_factory_fn: Incomplete | None = ...): ...

    orig_forward: Incomplete
    def call_module(self, m, forward, args, kwargs): ...
    prev_module: Incomplete
    def path_of_module(self, mod: torch.nn.Module) -> str: ...
    def proxy(self, node): ...
    meta_args: Incomplete
    patched_torch_methods: Incomplete
    orig_fns: Incomplete

    def trace(
        self, root, meta_args: Dict[str, torch.Tensor],
        concrete_args: Incomplete | None = ...): ...


def symbolic_trace(
    root: Union[torch.nn.Module, Callable[..., Any]], meta_args: Dict[str,
            torch.Tensor] = ..., concrete_args: Optional[Dict[str,
                Any]] = ...) -> torch.fx.GraphModule: ...
