from typing import Callable, Tuple, Union

from torch import Tensor as Tensor
from torch.utils._pytree import tree_flatten as tree_flatten
from torch.utils._pytree import tree_unflatten as tree_unflatten


in_dims_t = Union[int, Tuple]
out_dims_t = Union[int, Tuple[int, ...]]

def vmap(func: Callable, in_dims: in_dims_t = ..., out_dims: out_dims_t = ...) -> Callable: ...
