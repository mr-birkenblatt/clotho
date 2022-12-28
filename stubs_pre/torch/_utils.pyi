# pylint: disable=multiple-statements,unused-argument, invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias, unused-import
# pylint: disable=redefined-builtin,super-init-not-called, arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors, import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member, keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name, c-extension-no-member
# pylint: disable=protected-access,no-name-in-module, undefined-variable


from _typeshed import Incomplete


def annotate(ret, **kwargs): ...


class KeyErrorMessage(str):
    ...


class ExceptionWrapper:
    exc_type: Incomplete
    exc_msg: Incomplete
    where: Incomplete

    def __init__(
        self, exc_info: Incomplete | None = ..., where: str = ...) -> None: ...

    def reraise(self) -> None: ...


def get_current_device_index() -> int: ...


class _ClassPropertyDescriptor:
    fget: Incomplete
    def __init__(self, fget, fset: Incomplete | None = ...) -> None: ...
    def __get__(self, instance, owner: Incomplete | None = ...): ...


def classproperty(func): ...
