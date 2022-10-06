# Stubs for pandas._config.config (Python 3)
#
# NOTE: This dynamically typed stub was automatically generated by stubgen.
# pylint: disable=unused-import,unused-argument,invalid-name,redefined-builtin
# pylint: disable=too-few-public-methods,function-redefined
# pylint: disable=redefined-outer-name,too-many-ancestors,super-init-not-called
# pylint: disable=too-many-arguments

from collections import namedtuple
from typing import Any, Optional


DeprecatedOption = namedtuple("DeprecatedOption", "key msg rkey removal_ver")

RegisteredOption = namedtuple(
    "RegisteredOption", "key defval doc validator cb")


class OptionError(AttributeError, KeyError):
    ...


def get_default_val(pat: Any) -> Any:
    ...


class DictWrapper:
    def __init__(self, d: Any, prefix: str = ...) -> None:
        ...

    def __setattr__(self, key: Any, val: Any) -> None:
        ...

    def __getattr__(self, key: Any) -> Any:
        ...

    def __dir__(self) -> Any:
        ...


class CallableDynamicDoc:
    __doc_tmpl__: Any = ...
    __func__: Any = ...

    def __init__(self, func: Any, doc_tmpl: Any) -> None:
        ...

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        ...


get_option: Any
set_option: Any
reset_option: Any
describe_option: Any
options: Any


class option_context:
    ops: Any = ...

    def __init__(self, *args: Any) -> None:
        ...

    undo: Any = ...

    def __enter__(self) -> None:
        ...

    def __exit__(self, *args: Any) -> None:
        ...


def register_option(
        key: Any, defval: Any, doc: str = ...,
        validator: Optional[Any] = ...,
        cb: Optional[Any] = ...) -> None:
    ...


def deprecate_option(
        key: Any, msg: Optional[Any] = ...,
        rkey: Optional[Any] = ...,
        removal_ver: Optional[Any] = ...) -> None:
    ...


def pp_options_list(keys: Any, width: int = ..., _print: bool = ...) -> Any:
    ...


def config_prefix(prefix: Any) -> Any:
    ...


def is_type_factory(_type: Any) -> Any:
    ...


def is_instance_factory(_type: Any) -> Any:
    ...


def is_one_of_factory(legal_values: Any) -> Any:
    ...


is_int: Any
is_bool: Any
is_float: Any
is_str: Any
is_text: Any


def is_callable(obj: Any) -> Any:
    ...
