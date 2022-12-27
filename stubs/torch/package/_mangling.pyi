# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


class PackageMangler:
    def __init__(self) -> None: ...
    def mangle(self, name) -> str: ...
    def demangle(self, mangled: str) -> str: ...
    def parent_name(self): ...


def is_mangled(name: str) -> bool: ...


def demangle(name: str) -> str: ...


def get_mangle_prefix(name: str) -> str: ...
