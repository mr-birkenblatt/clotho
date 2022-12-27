# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


from typing import Dict, Iterable, Union

import torch
import torch.distributed.algorithms.model_averaging.averagers as averagers
from _typeshed import Incomplete


logger: Incomplete


class HierarchicalModelAverager(averagers.ModelAverager):
    period_process_group_dict: Incomplete
    warmup_steps: Incomplete

    def __init__(
        self, period_group_size_dict: Incomplete | None = ...,
        warmup_steps: int = ...,
        process_group: Incomplete | None = ...) -> None: ...

    def average_parameters(
        self, params: Union[Iterable[torch.nn.Parameter], Iterable[Dict[str,
        torch.nn.Parameter]]]): ...
