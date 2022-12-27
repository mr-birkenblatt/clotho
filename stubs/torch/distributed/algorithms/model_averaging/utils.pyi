# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


from typing import Dict, Iterable, Iterator, Union

import torch
from torch.distributed import group as group
from torch.distributed import ProcessGroup as ProcessGroup


def average_parameters(
    params: Iterator[torch.nn.Parameter], process_group: ProcessGroup): ...


def get_params_to_average(
    params: Union[Iterable[torch.nn.Parameter], Iterable[Dict[str,
    torch.nn.Parameter]]]): ...


def average_parameters_or_parameter_groups(
    params: Union[Iterable[torch.nn.Parameter], Iterable[Dict[str,
    torch.nn.Parameter]]], process_group: ProcessGroup): ...
