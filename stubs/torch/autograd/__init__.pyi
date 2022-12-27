# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


from typing import Optional

from torch.types import _TensorOrTensors

from .function import Function as Function
from .variable import Variable as Variable


def backward(
    tensors: _TensorOrTensors,
        grad_tensors: Optional[_TensorOrTensors] = ...,
        retain_graph: Optional[bool] = ..., create_graph: bool = ...,
        grad_variables: Optional[_TensorOrTensors] = ...,
        inputs: Optional[_TensorOrTensors] = ...) -> None: ...


# Names in __all__ with no definition:
#   grad_mode
