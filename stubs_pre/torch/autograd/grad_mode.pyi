# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access,no-name-in-module,undefined-variable


from typing import Any, Callable, TypeVar

from _typeshed import Incomplete


FuncType = Callable[..., Any]
F = TypeVar('F', bound=FuncType)


class _DecoratorContextManager:
    def __call__(self, func: F) -> F: ...
    def __enter__(self) -> None: ...

    def __exit__(
        self, exc_type: Any, exc_value: Any, traceback: Any) -> None: ...

    def clone(self): ...


class no_grad(_DecoratorContextManager):
    prev: bool
    def __init__(self) -> None: ...
    def __enter__(self) -> None: ...

    def __exit__(
        self, exc_type: Any, exc_value: Any, traceback: Any) -> None: ...


class enable_grad(_DecoratorContextManager):
    prev: Incomplete
    def __enter__(self) -> None: ...

    def __exit__(
        self, exc_type: Any, exc_value: Any, traceback: Any) -> None: ...


class set_grad_enabled(_DecoratorContextManager):
    prev: Incomplete
    mode: Incomplete
    def __init__(self, mode: bool) -> None: ...
    def __enter__(self) -> None: ...

    def __exit__(
        self, exc_type: Any, exc_value: Any, traceback: Any) -> None: ...

    def clone(self): ...


class inference_mode(_DecoratorContextManager):
    mode: Incomplete
    def __init__(self, mode: bool = ...) -> None: ...
    def __enter__(self) -> None: ...

    def __exit__(
        self, exc_type: Any, exc_value: Any, traceback: Any) -> None: ...

    def clone(self): ...
