# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


import torch

from .activation import ELU as ELU
from .activation import Hardswish as Hardswish


        LeakyReLU as LeakyReLU, ReLU6 as ReLU6, Sigmoid as Sigmoid,
        Softmax as Softmax
from .batchnorm import BatchNorm2d as BatchNorm2d
from .batchnorm import BatchNorm3d as BatchNorm3d
from .conv import Conv1d as Conv1d
from .conv import Conv2d as Conv2d
from .conv import Conv3d as Conv3d


        ConvTranspose1d as ConvTranspose1d,
        ConvTranspose2d as ConvTranspose2d,
        ConvTranspose3d as ConvTranspose3d, _ConvNd as _ConvNd
from .dropout import Dropout as Dropout
from .embedding_ops import Embedding as Embedding
from .embedding_ops import EmbeddingBag as EmbeddingBag
from .functional_modules import FXFloatFunctional as FXFloatFunctional


        FloatFunctional as FloatFunctional, QFunctional as QFunctional
from .linear import Linear as Linear
from .normalization import GroupNorm as GroupNorm


        InstanceNorm1d as InstanceNorm1d, InstanceNorm2d as InstanceNorm2d,
        InstanceNorm3d as InstanceNorm3d, LayerNorm as LayerNorm
from _typeshed import Incomplete
from torch.nn.modules.pooling import MaxPool2d as MaxPool2d


class Quantize(torch.nn.Module):
    scale: torch.Tensor
    zero_point: torch.Tensor
    dtype: Incomplete

    def __init__(
        self, scale, zero_point, dtype,
        factory_kwargs: Incomplete | None = ...) -> None: ...

    def forward(self, X): ...
    @staticmethod
    def from_float(mod): ...
    def extra_repr(self): ...


class DeQuantize(torch.nn.Module):
    def __init__(self) -> None: ...
    def forward(self, Xq): ...
    @staticmethod
    def from_float(mod): ...
