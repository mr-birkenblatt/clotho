# Stubs for pandas.io.sas.sas7bdat (Python 3)
#
# NOTE: This dynamically typed stub was automatically generated by stubgen.
# pylint: disable=unused-argument,redefined-outer-name,invalid-name
# pylint: disable=relative-beyond-top-level,arguments-differ
# pylint: disable=no-member,too-few-public-methods,keyword-arg-before-vararg
# pylint: disable=super-init-not-called,abstract-method,redefined-builtin
# pylint: disable=unused-import,useless-import-alias,signature-differs
# pylint: disable=blacklisted-name,c-extension-no-member
from typing import Any, Optional

from pandas.io.common import BaseIterator

class SAS7BDATReader(BaseIterator):
    index: Any = ...
    convert_dates: Any = ...
    blank_missing: Any = ...
    chunksize: Any = ...
    encoding: Any = ...
    convert_text: Any = ...
    convert_header_text: Any = ...
    default_encoding: str = ...
    compression: str = ...
    column_names_strings: Any = ...
    column_names: Any = ...
    column_formats: Any = ...
    columns: Any = ...
    handle: Any = ...

    def __init__(
            self, path_or_buf: Any, index: Optional[Any] = ...,
            convert_dates: bool = ..., blank_missing: bool = ...,
            chunksize: Optional[Any] = ..., encoding: Optional[Any] = ...,
            convert_text: bool = ...,
            convert_header_text: bool = ...) -> None:
        ...

    def column_data_lengths(self) -> Any:
        ...

    def column_data_offsets(self) -> Any:
        ...

    def column_types(self) -> Any:
        ...

    def close(self) -> None:
        ...

    def __next__(self) -> Any:
        ...

    def read(self, nrows: Optional[Any] = ...) -> Any:
        ...
