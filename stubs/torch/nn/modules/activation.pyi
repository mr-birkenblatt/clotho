# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


import torch

from .linear import as, NonDynamicallyQuantizableLinear


        NonDynamicallyQuantizableLinear
from _typeshed import Incomplete
from torch import Tensor as Tensor
from torch.nn.init import constant_ as constant_

from .module import Module as Module


        xavier_normal_ as xavier_normal_, xavier_uniform_ as xavier_uniform_
from typing import Optional, Tuple

from torch.nn.parameter import Parameter as Parameter


class Threshold(Module):
    __constants__: Incomplete
    threshold: float
    value: float
    inplace: bool

    def __init__(
        self, threshold: float, value: float, inplace: bool = ...) -> None: ...

    def forward(self, input: Tensor) -> Tensor: ...
    def extra_repr(self): ...


class ReLU(Module):
    __constants__: Incomplete
    inplace: bool
    def __init__(self, inplace: bool = ...) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...
    def extra_repr(self) -> str: ...


class RReLU(Module):
    __constants__: Incomplete
    lower: float
    upper: float
    inplace: bool

    def __init__(
        self, lower: float = ..., upper: float = ...,
        inplace: bool = ...) -> None: ...

    def forward(self, input: Tensor) -> Tensor: ...
    def extra_repr(self): ...


class Hardtanh(Module):
    __constants__: Incomplete
    min_val: float
    max_val: float
    inplace: bool

    def __init__(
        self, min_val: float = ..., max_val: float = ...,
        inplace: bool = ..., min_value: Optional[float] = ...,
        max_value: Optional[float] = ...) -> None: ...

    def forward(self, input: Tensor) -> Tensor: ...
    def extra_repr(self) -> str: ...


class ReLU6(Hardtanh):
    def __init__(self, inplace: bool = ...) -> None: ...
    def extra_repr(self) -> str: ...


class Sigmoid(Module):
    def forward(self, input: Tensor) -> Tensor: ...


class Hardsigmoid(Module):
    __constants__: Incomplete
    inplace: bool
    def __init__(self, inplace: bool = ...) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...


class Tanh(Module):
    def forward(self, input: Tensor) -> Tensor: ...


class SiLU(Module):
    __constants__: Incomplete
    inplace: bool
    def __init__(self, inplace: bool = ...) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...
    def extra_repr(self) -> str: ...


class Mish(Module):
    __constants__: Incomplete
    inplace: bool
    def __init__(self, inplace: bool = ...) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...
    def extra_repr(self) -> str: ...


class Hardswish(Module):
    __constants__: Incomplete
    inplace: bool
    def __init__(self, inplace: bool = ...) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...


class ELU(Module):
    __constants__: Incomplete
    alpha: float
    inplace: bool
    def __init__(self, alpha: float = ..., inplace: bool = ...) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...
    def extra_repr(self) -> str: ...


class CELU(Module):
    __constants__: Incomplete
    alpha: float
    inplace: bool
    def __init__(self, alpha: float = ..., inplace: bool = ...) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...
    def extra_repr(self) -> str: ...


class SELU(Module):
    __constants__: Incomplete
    inplace: bool
    def __init__(self, inplace: bool = ...) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...
    def extra_repr(self) -> str: ...


class GLU(Module):
    __constants__: Incomplete
    dim: int
    def __init__(self, dim: int = ...) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...
    def extra_repr(self) -> str: ...


class GELU(Module):
    __constants__: Incomplete
    approximate: str
    def __init__(self, approximate: str = ...) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...
    def extra_repr(self) -> str: ...


class Hardshrink(Module):
    __constants__: Incomplete
    lambd: float
    def __init__(self, lambd: float = ...) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...
    def extra_repr(self) -> str: ...


class LeakyReLU(Module):
    __constants__: Incomplete
    inplace: bool
    negative_slope: float

    def __init__(
        self, negative_slope: float = ..., inplace: bool = ...) -> None: ...

    def forward(self, input: Tensor) -> Tensor: ...
    def extra_repr(self) -> str: ...


class LogSigmoid(Module):
    def forward(self, input: Tensor) -> Tensor: ...


class Softplus(Module):
    __constants__: Incomplete
    beta: int
    threshold: int
    def __init__(self, beta: int = ..., threshold: int = ...) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...
    def extra_repr(self) -> str: ...


class Softshrink(Module):
    __constants__: Incomplete
    lambd: float
    def __init__(self, lambd: float = ...) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...
    def extra_repr(self) -> str: ...


class MultiheadAttention(Module):
    __constants__: Incomplete
    bias_k: Optional[torch.Tensor]
    bias_v: Optional[torch.Tensor]
    embed_dim: Incomplete
    kdim: Incomplete
    vdim: Incomplete
    num_heads: Incomplete
    dropout: Incomplete
    batch_first: Incomplete
    head_dim: Incomplete
    q_proj_weight: Incomplete
    k_proj_weight: Incomplete
    v_proj_weight: Incomplete
    in_proj_weight: Incomplete
    in_proj_bias: Incomplete
    out_proj: Incomplete
    add_zero_attn: Incomplete

    def __init__(
        self, embed_dim, num_heads, dropout: float = ..., bias: bool = ...,
        add_bias_kv: bool = ..., add_zero_attn: bool = ...,
        kdim: Incomplete | None = ..., vdim: Incomplete | None = ...,
        batch_first: bool = ..., device: Incomplete | None = ...,
        dtype: Incomplete | None = ...) -> None: ...

    def forward(
        self, query: Tensor, key: Tensor, value: Tensor,
        key_padding_mask: Optional[Tensor] = ..., need_weights: bool = ...,
        attn_mask: Optional[Tensor] = ...,
        average_attn_weights: bool = ...) -> Tuple[Tensor,
            Optional[Tensor]]: ...


class PReLU(Module):
    __constants__: Incomplete
    num_parameters: int
    weight: Incomplete

    def __init__(
        self, num_parameters: int = ..., init: float = ...,
        device: Incomplete | None = ...,
        dtype: Incomplete | None = ...) -> None: ...

    def forward(self, input: Tensor) -> Tensor: ...
    def extra_repr(self) -> str: ...


class Softsign(Module):
    def forward(self, input: Tensor) -> Tensor: ...


class Tanhshrink(Module):
    def forward(self, input: Tensor) -> Tensor: ...


class Softmin(Module):
    __constants__: Incomplete
    dim: Optional[int]
    def __init__(self, dim: Optional[int] = ...) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...
    def extra_repr(self): ...


class Softmax(Module):
    __constants__: Incomplete
    dim: Optional[int]
    def __init__(self, dim: Optional[int] = ...) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...
    def extra_repr(self) -> str: ...


class Softmax2d(Module):
    def forward(self, input: Tensor) -> Tensor: ...


class LogSoftmax(Module):
    __constants__: Incomplete
    dim: Optional[int]
    def __init__(self, dim: Optional[int] = ...) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...
    def extra_repr(self): ...
