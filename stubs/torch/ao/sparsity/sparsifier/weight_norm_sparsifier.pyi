# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


from typing import Tuple

from .base_sparsifier import BaseSparsifier as BaseSparsifier


class WeightNormSparsifier(BaseSparsifier):

    def __init__(
        self, sparsity_level: float = ..., sparse_block_shape: Tuple[int,
            int] = ..., zeros_per_block: int = ...): ...

    def update_mask(
        self, layer, sparsity_level, sparse_block_shape, zeros_per_block,
        **kwargs): ...
