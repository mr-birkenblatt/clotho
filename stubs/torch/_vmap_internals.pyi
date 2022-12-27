# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


from torch import Tensor as Tensor
from torch.utils._pytree import tree_flatten as tree_flatten


        tree_unflatten as tree_unflatten
from typing import Callable, Tuple, Union


in_dims_t = Union[int, Tuple]
out_dims_t = Union[int, Tuple[int, ...]]


def vmap(
    func: Callable, in_dims: in_dims_t = ...,
    out_dims: out_dims_t = ...) -> Callable: ...
