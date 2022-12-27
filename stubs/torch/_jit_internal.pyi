# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


import contextlib
import pickle
import weakref

import torch.distributed.rpc
from _typeshed import Incomplete
from torch._C._distributed_rpc import PyRRef as PyRRef
from torch._sources import fake_range as fake_range


        get_source_lines_and_file as get_source_lines_and_file,
        parse_def as parse_def
from typing import Any, Callable, Dict, List, Tuple, Type

from torch.distributed.rpc import RRef as RRef
from torch.futures import Future as Future


LockType: Type
boolean_dispatched: weakref.WeakKeyDictionary[Callable, Dict[str, Callable]]


def createResolutionCallbackFromEnv(lookup_base): ...


def createResolutionCallbackFromFrame(frames_up: int = ...): ...


def get_closure(fn): ...


def createResolutionCallbackFromClosure(fn): ...


def can_compile_class(cls) -> bool: ...


def get_callable_argument_names(fn) -> List[str]: ...


def get_annotation_str(annotation): ...


def get_type_hint_captures(fn): ...


def createResolutionCallbackForClassMethods(cls): ...


def boolean_dispatch(
    arg_name, arg_index, default, if_true, if_false, module_name,
    func_name): ...


class FunctionModifiers:
    UNUSED: str
    IGNORE: str
    EXPORT: str
    DEFAULT: str
    COPY_TO_SCRIPT_WRAPPER: str


def export(fn): ...


def unused(fn): ...


class _IgnoreContextManager(contextlib.AbstractContextManager):
    def __init__(self, **kwargs) -> None: ...

    def __exit__(
        self, exc_type: Any, exc_value: Any, traceback: Any) -> None: ...


def ignore(drop: bool = ..., **kwargs): ...


def module_has_exports(mod): ...


def should_drop(fn) -> bool: ...


def is_ignored_fn(fn) -> bool: ...


def is_static_fn(cls, fn) -> bool: ...


def get_static_fn(cls, fn): ...


def get_torchscript_modifier(fn): ...


def copy_torchscript_modifier(orig, new) -> None: ...


def get_overload_no_implementation_error_message(kind, obj): ...


def get_class_name_lineno(method) -> Tuple[str, int]: ...


def is_tuple(ann) -> bool: ...


def is_list(ann) -> bool: ...


def is_dict(ann) -> bool: ...


def is_union(ann): ...


def is_optional(ann): ...


def is_future(ann) -> bool: ...


def is_rref(ann) -> bool: ...


def is_rref_instance(obj) -> bool: ...


def is_final(ann) -> bool: ...


class BroadcastingListCls:
    def __getitem__(self, types) -> None: ...


BroadcastingList1: Incomplete


def is_scripting() -> bool: ...


def raise_error_container_parameter_missing(target_type) -> None: ...


def get_origin(target_type): ...


def get_args(target_type): ...


def check_args_exist(target_type) -> None: ...


def check_empty_containers(obj) -> None: ...


def container_checker(obj, target_type) -> bool: ...


class _TensorExtractor(pickle.Pickler):
    tensors: Incomplete

    def __init__(
        self, *args, tensors: List[torch.Tensor], **kwargs) -> None: ...

    def persistent_id(self, obj): ...
