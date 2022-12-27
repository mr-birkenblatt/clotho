# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


from _typeshed import Incomplete
from torch._prims.utils import as, ELEMENTWISE_TYPE_PROMOTION_KIND


        ELEMENTWISE_TYPE_PROMOTION_KIND, Number as Number,
        NumberType as NumberType, TensorLike as TensorLike,
        TensorLikeType as TensorLikeType
from typing import Callable, Sequence

from torch.utils._pytree import tree_flatten as tree_flatten


class elementwise_type_promotion_wrapper:
    type_promoting_arg_names: Incomplete
    type_promotion_kind: Incomplete

    def __init__(
        self, *, type_promotion_kind: ELEMENTWISE_TYPE_PROMOTION_KIND,
        type_promoting_args: Sequence[str] = ...) -> None: ...

    def __call__(self, fn: Callable) -> Callable: ...


def out_wrapper(fn: Callable) -> Callable: ...


def out_wrapper_multi(*out_names): ...
