# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


import torch
from _typeshed import Incomplete
from torch.ao.quantization import activation_is_memoryless, as


        activation_is_memoryless


class Linear(torch.nn.qat.Linear):

    def __init__(
        self, in_features, out_features, bias: bool = ...,
        qconfig: Incomplete | None = ..., device: Incomplete | None = ...,
        dtype: Incomplete | None = ...) -> None: ...
