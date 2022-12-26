from typing import Callable, Sequence

from _typeshed import Incomplete
from torch._prims.utils import (
    ELEMENTWISE_TYPE_PROMOTION_KIND as ELEMENTWISE_TYPE_PROMOTION_KIND,
)
from torch._prims.utils import Number as Number
from torch._prims.utils import NumberType as NumberType
from torch._prims.utils import TensorLike as TensorLike
from torch._prims.utils import TensorLikeType as TensorLikeType
from torch.utils._pytree import tree_flatten as tree_flatten


class elementwise_type_promotion_wrapper:
    type_promoting_arg_names: Incomplete
    type_promotion_kind: Incomplete
    def __init__(self, *, type_promotion_kind: ELEMENTWISE_TYPE_PROMOTION_KIND, type_promoting_args: Sequence[str] = ...) -> None: ...
    def __call__(self, fn: Callable) -> Callable: ...

def out_wrapper(fn: Callable) -> Callable: ...
def out_wrapper_multi(*out_names): ...
