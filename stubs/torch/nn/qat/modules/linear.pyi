# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


import torch.nn as nn
from _typeshed import Incomplete
from torch.nn.intrinsic import LinearReLU as LinearReLU
from torch.nn.utils.parametrize import is_parametrized as is_parametrized


        transfer_parametrizations_and_params as
        transfer_parametrizations_and_params,
        type_before_parametrizations as type_before_parametrizations


class Linear(nn.Linear):
    qconfig: Incomplete
    weight_fake_quant: Incomplete

    def __init__(
        self, in_features, out_features, bias: bool = ...,
        qconfig: Incomplete | None = ..., device: Incomplete | None = ...,
        dtype: Incomplete | None = ...) -> None: ...

    def forward(self, input): ...
    @classmethod
    def from_float(cls, mod): ...
    def to_float(self): ...
