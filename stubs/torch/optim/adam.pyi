# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


from typing import List, Optional

from _typeshed import Incomplete
from torch import Tensor as Tensor

from .optimizer import Optimizer as Optimizer


class Adam(Optimizer):

    def __init__(
        self, params, lr: float = ..., betas=..., eps: float = ...,
        weight_decay: int = ..., amsgrad: bool = ..., *,
        foreach: Optional[bool] = ..., maximize: bool = ...,
        capturable: bool = ...) -> None: ...

    def step(self, closure: Incomplete | None = ...): ...


def adam(
    params: List[Tensor], grads: List[Tensor], exp_avgs: List[Tensor],
        exp_avg_sqs: List[Tensor], max_exp_avg_sqs: List[Tensor],
        state_steps: List[Tensor], foreach: bool = ...,
        capturable: bool = ..., *, amsgrad: bool, beta1: float, beta2: float,
        lr: float, weight_decay: float, eps: float, maximize: bool): ...
