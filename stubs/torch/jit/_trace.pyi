# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


import torch
from _typeshed import Incomplete
from torch._jit_internal import (
    get_callable_argument_names as get_callable_argument_names,
)
from torch._jit_internal import is_scripting as is_scripting
from torch.autograd import function as function
from torch.jit._script import script as script
from torch.jit._script import ScriptModule as ScriptModule
from torch.nn import Module as Module
from torch.testing._comparison import default_tolerances as default_tolerances


class ONNXTracedModule(torch.nn.Module):
    inner: Incomplete
    strict: Incomplete

    def __init__(
        self, inner, strict: bool = ..., force_outplace: bool = ...,
        return_inputs: bool = ...,
        return_inputs_states: bool = ...) -> None: ...

    def forward(self, *args: torch.Tensor): ...


def verify(model, args, loss_fn=..., devices: Incomplete | None = ...): ...


def indent(s): ...


class TracingCheckError(Exception):
    message: str

    def __init__(
        self, graph_diff_error, tensor_compare_error,
        extra_msg: Incomplete | None = ...) -> None: ...


class TracerWarning(Warning):
    @staticmethod
    def ignore_lib_warnings() -> None: ...


def make_tuple(example_inputs): ...


def make_module(mod, _module_class, _compilation_unit): ...


def wrap_check_inputs(check_inputs): ...


def trace(
    func, example_inputs, optimize: Incomplete | None = ...,
        check_trace: bool = ..., check_inputs: Incomplete | None = ...,
        check_tolerance: float = ..., strict: bool = ...,
        _force_outplace: bool = ..., _module_class: Incomplete | None = ...,
        _compilation_unit=...): ...


def trace_module(
    mod, inputs, optimize: Incomplete | None = ..., check_trace: bool = ...,
        check_inputs: Incomplete | None = ..., check_tolerance: float = ...,
        strict: bool = ..., _force_outplace: bool = ...,
        _module_class: Incomplete | None = ..., _compilation_unit=...): ...


def is_tracing(): ...


class TracedModule(ScriptModule):

    def __init__(
        self, orig, id_set: Incomplete | None = ...,
        _compilation_unit: Incomplete | None = ...): ...

    def forward(self, *args, **kwargs) -> None: ...
    def __getattr__(self, attr): ...
    def __setattr__(self, attr, value): ...
    def extra_repr(self): ...


class TopLevelTracedModule(TracedModule):
    forward: Incomplete
