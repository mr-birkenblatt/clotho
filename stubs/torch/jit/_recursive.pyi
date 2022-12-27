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
from torch._sources import fake_range as fake_range
from torch.jit._check import (
    AttributeTypeIsSupportedChecker as AttributeTypeIsSupportedChecker,
)
from torch.jit.frontend import get_class_properties as get_class_properties


        get_default_args as get_default_args,
        get_jit_class_def as get_jit_class_def, get_jit_def as get_jit_def
from typing import Dict, List, NamedTuple, Set, Type

from torch.nn import Module as Module


class ScriptMethodStub(NamedTuple):
    resolution_callback: Incomplete
    def_: Incomplete
    original_method: Incomplete


class PropertyStub(NamedTuple):
    resolution_callback: Incomplete
    def_: Incomplete
ignored_attributes: Incomplete


def make_stub(func, name): ...


def make_stub_from_method(nn_module, method_name): ...


def make_stubs_from_exported_methods(mod): ...


def jit_ignored_properties(module): ...


class SourceContext(torch._C._jit_tree_views.SourceRangeFactory):

    def __init__(
        self, source, filename, file_lineno,
        leading_whitespace_len) -> None: ...


def infer_concrete_type_builder(nn_module, share_types: bool = ...): ...


class ConcreteTypeStore:
    type_store: Dict[Type[Module], List[torch._C.ConcreteModuleType]]
    methods_compiled: Set[torch._C.ConcreteModuleType]
    def __init__(self) -> None: ...
    def get_or_create_concrete_type(self, nn_module): ...


concrete_type_store: Incomplete


def create_methods_and_properties_from_stubs(
    concrete_type, method_stubs, property_stubs) -> None: ...


def create_hooks_from_stubs(
    concrete_type, hook_stubs, pre_hook_stubs) -> None: ...


def get_module_concrete_type(nn_module, share_types: bool = ...): ...


def create_script_class(obj): ...


def create_script_module(
    nn_module, stubs_fn, share_types: bool = ..., is_tracing: bool = ...): ...


def create_script_module_impl(nn_module, concrete_type, stubs_fn): ...


def script_model_defines_attr(script_model, attr): ...


def add_python_attr_to_scripted_model(script_model, orig, attr) -> None: ...


def get_overload_annotations(mod, jit_ignored_properties): ...


def get_overload_name_mapping(overload_info): ...


def make_stubs_for_overloads(overload_info): ...


def check_module_initialized(mod) -> None: ...


def infer_methods_to_compile(nn_module): ...


def get_hook_stubs(nn_module): ...


def get_property_stubs(nn_module): ...


def interface_script(mod_interface, nn_module): ...


def try_compile_fn(fn, loc): ...


def wrap_cpp_class(cpp_class): ...


def wrap_cpp_module(cpp_module): ...


def compile_unbound_method(concrete_type, fn): ...


def lazy_bind(concrete_type, unbound_method): ...
