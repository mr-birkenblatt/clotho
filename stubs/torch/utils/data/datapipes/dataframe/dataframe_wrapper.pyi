# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level
from collections.abc import Generator

from _typeshed import Incomplete


class PandasWrapper:
    @classmethod
    def create_dataframe(cls, data, columns): ...
    @classmethod
    def is_dataframe(cls, data): ...
    @classmethod
    def is_column(cls, data): ...
    @classmethod
    def iterate(cls, data) -> Generator[Incomplete, None, None]: ...
    @classmethod
    def concat(cls, buffer): ...
    @classmethod
    def get_item(cls, data, idx): ...
    @classmethod
    def get_len(cls, df): ...


default_wrapper = PandasWrapper


def get_df_wrapper(): ...


def set_df_wrapper(wrapper) -> None: ...


def create_dataframe(data, columns: Incomplete | None = ...): ...


def is_dataframe(data): ...


def is_column(data): ...


def concat(buffer): ...


def iterate(data): ...


def get_item(data, idx): ...


def get_len(df): ...
