# Stubs for pandas.util._decorators (Python 3)
#
# NOTE: This dynamically typed stub was automatically generated by stubgen.
# pylint: disable=unused-argument,redefined-outer-name,invalid-name
# pylint: disable=relative-beyond-top-level,arguments-differ
# pylint: disable=no-member,too-few-public-methods,keyword-arg-before-vararg
# pylint: disable=super-init-not-called,abstract-method,redefined-builtin
# pylint: disable=unused-import,useless-import-alias,signature-differs
# pylint: disable=blacklisted-name,c-extension-no-member,too-many-ancestors

from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union


def deprecate(
        name: str, alternative: Callable, version: str,
        alt_name: Optional[str] = ...,
        klass: Optional[Type[Warning]] = ...,
        stacklevel: int = ...,
        msg: Optional[str] = ...) -> Callable:
    ...


def deprecate_kwarg(
        old_arg_name: str, new_arg_name: Optional[str],
        mapping: Optional[Union[Dict, Callable[[Any], Any]]] = ...,
        stacklevel: int = ...) -> Callable:
    ...


def rewrite_axis_style_signature(
        name: str, extra_params: List[Tuple[str, Any]]) -> Callable:
    ...


class Substitution:
    params: Any = ...

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        ...

    def __call__(self, func: Callable) -> Callable:
        ...

    def update(self, *args: Any, **kwargs: Any) -> None:
        ...


class Appender:
    addendum: Any = ...
    join: Any = ...

    def __init__(
            self, addendum: Optional[str], join: str = ...,
            indents: int = ...) -> None:
        ...

    def __call__(self, func: Callable) -> Callable:
        ...


def indent(text: Optional[str], indents: int = ...) -> str:
    ...
