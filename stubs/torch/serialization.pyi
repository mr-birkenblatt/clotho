# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


import os
from collections.abc import Generator
from typing import BinaryIO, IO, Union

import torch
from _typeshed import Incomplete
from torch._sources import (
    get_source_lines_and_file as get_source_lines_and_file,
)
from torch.types import Storage as Storage


DEFAULT_PROTOCOL: int
LONG_SIZE: Incomplete
INT_SIZE: Incomplete
SHORT_SIZE: Incomplete
MAGIC_NUMBER: int
PROTOCOL_VERSION: int
STORAGE_KEY_SEPARATOR: str


class SourceChangeWarning(Warning):
    ...


def mkdtemp() -> Generator[Incomplete, None, None]: ...


def register_package(priority, tagger, deserializer) -> None: ...


def check_module_version_greater_or_equal(
    module, req_version_tuple, error_if_malformed: bool = ...): ...


def validate_cuda_device(location): ...


def location_tag(
    storage: Union[Storage, torch.storage._TypedStorage,
    torch._UntypedStorage]): ...


def default_restore_location(storage, location): ...


def normalize_storage_type(storage_type): ...


def storage_to_tensor_type(storage): ...


class _opener:
    file_like: Incomplete
    def __init__(self, file_like) -> None: ...
    def __enter__(self): ...
    def __exit__(self, *args) -> None: ...


class _open_file(_opener):
    def __init__(self, name, mode) -> None: ...
    def __exit__(self, *args) -> None: ...


class _open_buffer_reader(_opener):
    def __init__(self, buffer) -> None: ...


class _open_buffer_writer(_opener):
    def __exit__(self, *args) -> None: ...


class _open_zipfile_reader(_opener):
    def __init__(self, name_or_buffer) -> None: ...


class _open_zipfile_writer_file(_opener):
    def __init__(self, name) -> None: ...
    def __exit__(self, *args) -> None: ...


class _open_zipfile_writer_buffer(_opener):
    buffer: Incomplete
    def __init__(self, buffer) -> None: ...
    def __exit__(self, *args) -> None: ...


def save(
    obj, f: Union[str, os.PathLike, BinaryIO, IO[bytes]], pickle_module=...,
    pickle_protocol=...,
    _use_new_zipfile_serialization: bool = ...) -> None: ...


def load(
    f, map_location: Incomplete | None = ..., pickle_module=...,
    **pickle_load_args): ...


class StorageType:
    dtype: Incomplete
    def __init__(self, name) -> None: ...
