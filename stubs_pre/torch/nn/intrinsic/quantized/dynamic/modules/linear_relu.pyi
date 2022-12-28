# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access,no-name-in-module,undefined-variable


import torch
import torch.nn.quantized.dynamic as nnqd


class LinearReLU(nnqd.Linear):

    def __init__(
        self, in_features, out_features, bias: bool = ...,
        dtype=...) -> None: ...

    def forward(self, x: torch.Tensor) -> torch.Tensor: ...
    @classmethod
    def from_float(cls, mod): ...
    @classmethod
    def from_reference(cls, ref_qlinear_relu): ...
