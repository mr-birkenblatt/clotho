# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access,no-name-in-module,undefined-variable


from typing import NamedTuple

from _typeshed import Incomplete


class Entry(NamedTuple):
    version: Incomplete
    hash: Incomplete


def update_hash(seed, value): ...


def hash_source_files(hash_value, source_files): ...


def hash_build_arguments(hash_value, build_arguments): ...


class ExtensionVersioner:
    entries: Incomplete
    def __init__(self) -> None: ...
    def get_version(self, name): ...

    def bump_version_if_changed(
        self, name, source_files, build_arguments, build_directory,
        with_cuda, is_python_module, is_standalone): ...
