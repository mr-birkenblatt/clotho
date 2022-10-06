# pylint: disable=import-error,relative-beyond-top-level,unused-import
# pylint: disable=useless-import-alias,multiple-statements,no-name-in-module
# pylint: disable=unused-argument,invalid-name,super-init-not-called
import abc
from abc import ABC, abstractmethod
from typing import Any

from _typeshed import Incomplete


class BaseTokenManager(ABC, metaclass=abc.ABCMeta):
    def __init__(self) -> None: ...

    @property
    def reddit(self) -> None: ...

    @reddit.setter
    def reddit(self, value: Any) -> None: ...

    @abstractmethod
    def post_refresh_callback(self, authorizer: Any) -> None: ...

    @abstractmethod
    def pre_refresh_callback(self, authorizer: Any) -> None: ...


class FileTokenManager(BaseTokenManager):
    def __init__(self, filename: str) -> None: ...

    def post_refresh_callback(self, authorizer: Any) -> None: ...

    def pre_refresh_callback(self, authorizer: Any) -> None: ...


class SQLiteTokenManager(BaseTokenManager):
    key: Incomplete

    def __init__(self, *, database: Any, key: Any) -> None: ...

    def is_registered(self) -> None: ...

    def post_refresh_callback(self, authorizer: Any) -> None: ...

    def pre_refresh_callback(self, authorizer: Any) -> None: ...

    def register(self, refresh_token: Any) -> None: ...
