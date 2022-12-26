from typing import Optional, Tuple

from torch import Tensor


def svd_lowrank(A: Tensor, q: Optional[int] = ..., niter: Optional[int] = ..., M: Optional[Tensor] = ...) -> Tuple[Tensor, Tensor, Tensor]: ...
def pca_lowrank(A: Tensor, q: Optional[int] = ..., center: bool = ..., niter: int = ...) -> Tuple[Tensor, Tensor, Tensor]: ...
