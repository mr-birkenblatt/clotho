# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


def register_version(domain: str, version: int): ...


def register_ops_helper(domain: str, version: int, iter_version: int): ...


def register_ops_in_version(domain: str, version: int): ...


def get_ops_in_version(version: int): ...


def is_registered_version(domain: str, version: int): ...


def register_op(opname, op, domain, version) -> None: ...


def is_registered_op(opname: str, domain: str, version: int): ...


def unregister_op(opname: str, domain: str, version: int): ...


def get_op_supported_version(opname: str, domain: str, version: int): ...


def get_registered_op(
    opname: str, domain: str, version: int) -> _SymbolicFunction: ...


class UnsupportedOperatorError(RuntimeError):
    def __init__(self, domain: str, opname: str, version: int) -> None: ...
