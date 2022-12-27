# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


from typing import Optional


def get_enum(reduction: str) -> int: ...


def legacy_get_string(
    size_average: Optional[bool], reduce: Optional[bool],
    emit_warning: bool = ...) -> str: ...


def legacy_get_enum(
    size_average: Optional[bool], reduce: Optional[bool],
    emit_warning: bool = ...) -> int: ...
