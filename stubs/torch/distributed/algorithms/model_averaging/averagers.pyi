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
from typing import Dict, Iterable, Union

import torch
from _typeshed import Incomplete


class ModelAverager(ABC, metaclass=abc.ABCMeta):
    process_group: Incomplete
    step: int
    def __init__(self, process_group: Incomplete | None = ...) -> None: ...
    @abstractmethod
    def average_parameters(self, params): ...


class PeriodicModelAverager(ModelAverager):
    warmup_steps: Incomplete
    period: Incomplete

    def __init__(
        self, period, warmup_steps: int = ...,
        process_group: Incomplete | None = ...) -> None: ...

    def average_parameters(
        self, params: Union[Iterable[torch.nn.Parameter], Iterable[Dict[str,
                                torch.nn.Parameter]]]): ...
