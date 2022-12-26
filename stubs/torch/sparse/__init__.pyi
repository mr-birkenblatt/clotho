from typing import List, Optional, Tuple, Union

from _typeshed import Incomplete
from torch import Tensor
from torch.types import _dtype as DType


DimOrDims = Optional[Union[int, Tuple[int], List[int]]]
addmm: Incomplete
mm: Incomplete

def sum(input: Tensor, dim: DimOrDims = ..., dtype: Optional[DType] = ...) -> Tensor: ...

softmax: Incomplete
log_softmax: Incomplete
