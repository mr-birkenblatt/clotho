from typing import Tuple

from .base_sparsifier import BaseSparsifier as BaseSparsifier


class WeightNormSparsifier(BaseSparsifier):
    def __init__(self, sparsity_level: float = ..., sparse_block_shape: Tuple[int, int] = ..., zeros_per_block: int = ...): ...
    def update_mask(self, layer, sparsity_level, sparse_block_shape, zeros_per_block, **kwargs): ...
