# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


from typing import List, Optional, Tuple, Union

from _typeshed import Incomplete
from torch import Tensor
from torch.types import _dtype as DType


DimOrDims = Optional[Union[int, Tuple[int], List[int]]]
addmm: Incomplete
mm: Incomplete


def sum(
    input: Tensor, dim: DimOrDims = ...,
        dtype: Optional[DType] = ...) -> Tensor: ...


softmax: Incomplete
log_softmax: Incomplete
