# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access,no-name-in-module,undefined-variable


import pickle

from _typeshed import Incomplete


class FakeObject:
    module: Incomplete
    name: Incomplete
    args: Incomplete
    state: Incomplete
    def __init__(self, module, name, args) -> None: ...

    @staticmethod
    def pp_format(
        printer, obj, stream, indent, allowance, context, level) -> None: ...


class FakeClass:
    module: Incomplete
    name: Incomplete
    __new__: Incomplete
    def __init__(self, module, name) -> None: ...
    def __call__(self, *args): ...
    def fake_new(self, *args): ...


class DumpUnpickler(pickle._Unpickler):
    catch_invalid_utf8: Incomplete

    def __init__(
        self, file, *, catch_invalid_utf8: bool = ..., **kwargs) -> None: ...

    def find_class(self, module, name): ...
    def persistent_load(self, pid): ...
    dispatch: Incomplete
    def load_binunicode(self) -> None: ...
    @classmethod
    def dump(cls, in_stream, out_stream): ...


def main(argv, output_stream: Incomplete | None = ...): ...
