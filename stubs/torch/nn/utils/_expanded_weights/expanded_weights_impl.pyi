# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


from typing import Callable, Dict

import torch
from _typeshed import Incomplete


HANDLED_FUNCTIONS: Dict[Callable, torch.autograd.Function]


def implements_per_sample_grads(torch_function): ...


class ExpandedWeight(torch.Tensor):
    batch_size: Incomplete
    orig_weight: Incomplete
    def __init__(self, orig_weight, batch_size) -> None: ...
    handled_functions: Incomplete
    def __new__(cls, orig_weight, _): ...

    @classmethod
    def __torch_function__(
        cls, func, _, args=..., kwargs: Incomplete | None = ...): ...

    @property
    def dtype(self): ...
    @property
    def shape(self): ...
