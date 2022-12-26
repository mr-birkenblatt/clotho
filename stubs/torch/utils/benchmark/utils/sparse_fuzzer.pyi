from typing import Optional, Tuple, Union

from torch.utils.benchmark import FuzzedTensor as FuzzedTensor


class FuzzedSparseTensor(FuzzedTensor):
    def __init__(self, name: str, size: Tuple[Union[str, int], ...], min_elements: Optional[int] = ..., max_elements: Optional[int] = ..., dim_parameter: Optional[str] = ..., sparse_dim: Optional[str] = ..., nnz: Optional[str] = ..., density: Optional[str] = ..., coalesced: Optional[str] = ..., dtype=..., cuda: bool = ...) -> None: ...
    @staticmethod
    def sparse_tensor_constructor(size, dtype, sparse_dim, nnz, is_coalesced): ...
