# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


from io import IOBase
from typing import Iterable, List, Optional, Tuple, Union

from _typeshed import Incomplete


def match_masks(name: str, masks: Union[str, List[str]]) -> bool: ...


def get_file_pathnames_from_root(
    root: str, masks: Union[str, List[str]], recursive: bool = ...,
        abspath: bool = ...,
        non_deterministic: bool = ...) -> Iterable[str]: ...


def get_file_binaries_from_pathnames(
    pathnames: Iterable, mode: str, encoding: Optional[str] = ...): ...


def validate_pathname_binary_tuple(data: Tuple[str, IOBase]): ...


class StreamWrapper:
    file_obj: Incomplete
    def __init__(self, file_obj) -> None: ...
    def __getattr__(self, name): ...
    def __dir__(self): ...
    def __del__(self) -> None: ...
    def __iter__(self): ...
    def __next__(self): ...
