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
    def __init__(self, period, warmup_steps: int = ..., process_group: Incomplete | None = ...) -> None: ...
    def average_parameters(self, params: Union[Iterable[torch.nn.Parameter], Iterable[Dict[str, torch.nn.Parameter]]]): ...
