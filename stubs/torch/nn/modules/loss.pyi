# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


from typing import Callable, Optional

from _typeshed import Incomplete
from torch import Tensor as Tensor

from .distance import PairwiseDistance as PairwiseDistance
from .module import Module as Module


class _Loss(Module):
    reduction: str

    def __init__(
        self, size_average: Incomplete | None = ...,
        reduce: Incomplete | None = ..., reduction: str = ...) -> None: ...


class _WeightedLoss(_Loss):
    weight: Incomplete

    def __init__(
        self, weight: Optional[Tensor] = ...,
        size_average: Incomplete | None = ...,
        reduce: Incomplete | None = ..., reduction: str = ...) -> None: ...


class L1Loss(_Loss):
    __constants__: Incomplete

    def __init__(
        self, size_average: Incomplete | None = ...,
        reduce: Incomplete | None = ..., reduction: str = ...) -> None: ...

    def forward(self, input: Tensor, target: Tensor) -> Tensor: ...


class NLLLoss(_WeightedLoss):
    __constants__: Incomplete
    ignore_index: int

    def __init__(
        self, weight: Optional[Tensor] = ...,
        size_average: Incomplete | None = ..., ignore_index: int = ...,
        reduce: Incomplete | None = ..., reduction: str = ...) -> None: ...

    def forward(self, input: Tensor, target: Tensor) -> Tensor: ...


class NLLLoss2d(NLLLoss):

    def __init__(
        self, weight: Optional[Tensor] = ...,
        size_average: Incomplete | None = ..., ignore_index: int = ...,
        reduce: Incomplete | None = ..., reduction: str = ...) -> None: ...


class PoissonNLLLoss(_Loss):
    __constants__: Incomplete
    log_input: bool
    full: bool
    eps: float

    def __init__(
        self, log_input: bool = ..., full: bool = ...,
        size_average: Incomplete | None = ..., eps: float = ...,
        reduce: Incomplete | None = ..., reduction: str = ...) -> None: ...

    def forward(self, log_input: Tensor, target: Tensor) -> Tensor: ...


class GaussianNLLLoss(_Loss):
    __constants__: Incomplete
    full: bool
    eps: float

    def __init__(
        self, *, full: bool = ..., eps: float = ...,
        reduction: str = ...) -> None: ...

    def forward(
        self, input: Tensor, target: Tensor, var: Tensor) -> Tensor: ...


class KLDivLoss(_Loss):
    __constants__: Incomplete
    log_target: Incomplete

    def __init__(
        self, size_average: Incomplete | None = ...,
        reduce: Incomplete | None = ..., reduction: str = ...,
        log_target: bool = ...) -> None: ...

    def forward(self, input: Tensor, target: Tensor) -> Tensor: ...


class MSELoss(_Loss):
    __constants__: Incomplete

    def __init__(
        self, size_average: Incomplete | None = ...,
        reduce: Incomplete | None = ..., reduction: str = ...) -> None: ...

    def forward(self, input: Tensor, target: Tensor) -> Tensor: ...


class BCELoss(_WeightedLoss):
    __constants__: Incomplete

    def __init__(
        self, weight: Optional[Tensor] = ...,
        size_average: Incomplete | None = ...,
        reduce: Incomplete | None = ..., reduction: str = ...) -> None: ...

    def forward(self, input: Tensor, target: Tensor) -> Tensor: ...


class BCEWithLogitsLoss(_Loss):
    weight: Incomplete
    pos_weight: Incomplete

    def __init__(
        self, weight: Optional[Tensor] = ...,
        size_average: Incomplete | None = ...,
        reduce: Incomplete | None = ..., reduction: str = ...,
        pos_weight: Optional[Tensor] = ...) -> None: ...

    def forward(self, input: Tensor, target: Tensor) -> Tensor: ...


class HingeEmbeddingLoss(_Loss):
    __constants__: Incomplete
    margin: float

    def __init__(
        self, margin: float = ..., size_average: Incomplete | None = ...,
        reduce: Incomplete | None = ..., reduction: str = ...) -> None: ...

    def forward(self, input: Tensor, target: Tensor) -> Tensor: ...


class MultiLabelMarginLoss(_Loss):
    __constants__: Incomplete

    def __init__(
        self, size_average: Incomplete | None = ...,
        reduce: Incomplete | None = ..., reduction: str = ...) -> None: ...

    def forward(self, input: Tensor, target: Tensor) -> Tensor: ...


class SmoothL1Loss(_Loss):
    __constants__: Incomplete
    beta: Incomplete

    def __init__(
        self, size_average: Incomplete | None = ...,
        reduce: Incomplete | None = ..., reduction: str = ...,
        beta: float = ...) -> None: ...

    def forward(self, input: Tensor, target: Tensor) -> Tensor: ...


class HuberLoss(_Loss):
    __constants__: Incomplete
    delta: Incomplete
    def __init__(self, reduction: str = ..., delta: float = ...) -> None: ...
    def forward(self, input: Tensor, target: Tensor) -> Tensor: ...


class SoftMarginLoss(_Loss):
    __constants__: Incomplete

    def __init__(
        self, size_average: Incomplete | None = ...,
        reduce: Incomplete | None = ..., reduction: str = ...) -> None: ...

    def forward(self, input: Tensor, target: Tensor) -> Tensor: ...


class CrossEntropyLoss(_WeightedLoss):
    __constants__: Incomplete
    ignore_index: int
    label_smoothing: float

    def __init__(
        self, weight: Optional[Tensor] = ...,
        size_average: Incomplete | None = ..., ignore_index: int = ...,
        reduce: Incomplete | None = ..., reduction: str = ...,
        label_smoothing: float = ...) -> None: ...

    def forward(self, input: Tensor, target: Tensor) -> Tensor: ...


class MultiLabelSoftMarginLoss(_WeightedLoss):
    __constants__: Incomplete

    def __init__(
        self, weight: Optional[Tensor] = ...,
        size_average: Incomplete | None = ...,
        reduce: Incomplete | None = ..., reduction: str = ...) -> None: ...

    def forward(self, input: Tensor, target: Tensor) -> Tensor: ...


class CosineEmbeddingLoss(_Loss):
    __constants__: Incomplete
    margin: float

    def __init__(
        self, margin: float = ..., size_average: Incomplete | None = ...,
        reduce: Incomplete | None = ..., reduction: str = ...) -> None: ...

    def forward(
        self, input1: Tensor, input2: Tensor, target: Tensor) -> Tensor: ...


class MarginRankingLoss(_Loss):
    __constants__: Incomplete
    margin: float

    def __init__(
        self, margin: float = ..., size_average: Incomplete | None = ...,
        reduce: Incomplete | None = ..., reduction: str = ...) -> None: ...

    def forward(
        self, input1: Tensor, input2: Tensor, target: Tensor) -> Tensor: ...


class MultiMarginLoss(_WeightedLoss):
    __constants__: Incomplete
    margin: float
    p: int

    def __init__(
        self, p: int = ..., margin: float = ...,
        weight: Optional[Tensor] = ...,
        size_average: Incomplete | None = ...,
        reduce: Incomplete | None = ..., reduction: str = ...) -> None: ...

    def forward(self, input: Tensor, target: Tensor) -> Tensor: ...


class TripletMarginLoss(_Loss):
    __constants__: Incomplete
    margin: float
    p: float
    eps: float
    swap: bool

    def __init__(
        self, margin: float = ..., p: float = ..., eps: float = ...,
        swap: bool = ..., size_average: Incomplete | None = ...,
        reduce: Incomplete | None = ..., reduction: str = ...) -> None: ...

    def forward(
        self, anchor: Tensor, positive: Tensor,
        negative: Tensor) -> Tensor: ...


class TripletMarginWithDistanceLoss(_Loss):
    __constants__: Incomplete
    margin: float
    swap: bool
    distance_function: Incomplete

    def __init__(
        self, *, distance_function: Optional[Callable[[Tensor, Tensor],
                        Tensor]] = ..., margin: float = ...,
        swap: bool = ..., reduction: str = ...) -> None: ...

    def forward(
        self, anchor: Tensor, positive: Tensor,
        negative: Tensor) -> Tensor: ...


class CTCLoss(_Loss):
    __constants__: Incomplete
    blank: int
    zero_infinity: bool

    def __init__(
        self, blank: int = ..., reduction: str = ...,
        zero_infinity: bool = ...) -> None: ...

    def forward(
        self, log_probs: Tensor, targets: Tensor, input_lengths: Tensor,
        target_lengths: Tensor) -> Tensor: ...
