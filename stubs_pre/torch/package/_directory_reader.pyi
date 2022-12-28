# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access,no-name-in-module,undefined-variable


from _typeshed import Incomplete
from torch.types import Storage as Storage


class _HasStorage:
    def __init__(self, storage) -> None: ...
    def storage(self): ...


class DirectoryReader:
    directory: Incomplete
    def __init__(self, directory) -> None: ...
    def get_record(self, name): ...
    def get_storage_from_record(self, name, numel, dtype): ...
    def has_record(self, path): ...
    def get_all_records(self): ...
