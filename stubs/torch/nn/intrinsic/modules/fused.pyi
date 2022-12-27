# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


import torch
from torch.nn import BatchNorm1d as BatchNorm1d
from torch.nn import BatchNorm2d as BatchNorm2d
from torch.nn import BatchNorm3d as BatchNorm3d
from torch.nn import Conv1d as Conv1d
from torch.nn import Conv2d as Conv2d
from torch.nn import Conv3d as Conv3d
from torch.nn import Linear as Linear
from torch.nn import ReLU as ReLU
from torch.nn.utils.parametrize import (
    type_before_parametrizations as type_before_parametrizations,
)


class _FusedModule(torch.nn.Sequential):
    ...


class ConvReLU1d(_FusedModule):
    def __init__(self, conv, relu) -> None: ...


class ConvReLU2d(_FusedModule):
    def __init__(self, conv, relu) -> None: ...


class ConvReLU3d(_FusedModule):
    def __init__(self, conv, relu) -> None: ...


class LinearReLU(_FusedModule):
    def __init__(self, linear, relu) -> None: ...


class ConvBn1d(_FusedModule):
    def __init__(self, conv, bn) -> None: ...


class ConvBn2d(_FusedModule):
    def __init__(self, conv, bn) -> None: ...


class ConvBnReLU1d(_FusedModule):
    def __init__(self, conv, bn, relu) -> None: ...


class ConvBnReLU2d(_FusedModule):
    def __init__(self, conv, bn, relu) -> None: ...


class ConvBn3d(_FusedModule):
    def __init__(self, conv, bn) -> None: ...


class ConvBnReLU3d(_FusedModule):
    def __init__(self, conv, bn, relu) -> None: ...


class BNReLU2d(_FusedModule):
    def __init__(self, batch_norm, relu) -> None: ...


class BNReLU3d(_FusedModule):
    def __init__(self, batch_norm, relu) -> None: ...


class LinearBn1d(_FusedModule):
    def __init__(self, linear, bn) -> None: ...
