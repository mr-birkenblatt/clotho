# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access,no-name-in-module,undefined-variable


import abc
from abc import ABC, abstractmethod
from types import TracebackType
from typing import Any, List, NamedTuple, Optional, Type

import torch


class JoinHook:
    def main_hook(self) -> None: ...
    def post_hook(self, is_last_joiner: bool) -> None: ...


class Joinable(ABC, metaclass=abc.ABCMeta):
    @abstractmethod
    def __init__(self): ...
    @abstractmethod
    def join_hook(self, **kwargs) -> JoinHook: ...
    @property
    @abstractmethod
    def join_device(self) -> torch.device: ...
    @property
    @abstractmethod
    def join_process_group(self) -> Any: ...


class _JoinConfig(NamedTuple):
    enable: bool
    throw_on_early_termination: bool
    is_first_joinable: bool
    @staticmethod
    def construct_disabled_join_config(): ...


class Join:

    def __init__(
        self, joinables: List[Joinable], enable: bool = ...,
        throw_on_early_termination: bool = ..., **kwargs) -> None: ...

    def __enter__(self) -> None: ...

    def __exit__(
        self, type: Optional[Type[BaseException]],
        value: Optional[BaseException],
        traceback: Optional[TracebackType]): ...

    @staticmethod
    def notify_join_context(joinable: Joinable): ...
