# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access,no-name-in-module,undefined-variable


from typing import List, Optional

from _typeshed import Incomplete
from torch import Tensor as Tensor

from .optimizer import Optimizer as Optimizer


class RMSprop(Optimizer):

    def __init__(
        self, params, lr: float = ..., alpha: float = ..., eps: float = ...,
        weight_decay: int = ..., momentum: int = ..., centered: bool = ...,
        foreach: Optional[bool] = ...) -> None: ...

    def step(self, closure: Incomplete | None = ...): ...


def rmsprop(
    params: List[Tensor], grads: List[Tensor], square_avgs: List[Tensor],
    grad_avgs: List[Tensor], momentum_buffer_list: List[Tensor],
    foreach: bool = ..., *, lr: float, alpha: float, eps: float,
    weight_decay: float, momentum: float, centered: bool): ...
