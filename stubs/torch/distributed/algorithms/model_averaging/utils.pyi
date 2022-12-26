from typing import Dict, Iterable, Iterator, Union

import torch
from torch.distributed import group as group
from torch.distributed import ProcessGroup as ProcessGroup


def average_parameters(params: Iterator[torch.nn.Parameter], process_group: ProcessGroup): ...
def get_params_to_average(params: Union[Iterable[torch.nn.Parameter], Iterable[Dict[str, torch.nn.Parameter]]]): ...
def average_parameters_or_parameter_groups(params: Union[Iterable[torch.nn.Parameter], Iterable[Dict[str, torch.nn.Parameter]]], process_group: ProcessGroup): ...
