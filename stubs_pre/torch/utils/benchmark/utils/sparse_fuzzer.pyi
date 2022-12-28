# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access,no-name-in-module,undefined-variable


from typing import Optional, Tuple, Union

from torch.utils.benchmark import FuzzedTensor as FuzzedTensor


class FuzzedSparseTensor(FuzzedTensor):

    def __init__(
        self, name: str, size: Tuple[Union[str, int], ...],
        min_elements: Optional[int] = ..., max_elements: Optional[int] = ...,
        dim_parameter: Optional[str] = ..., sparse_dim: Optional[str] = ...,
        nnz: Optional[str] = ..., density: Optional[str] = ...,
        coalesced: Optional[
                str] = ..., dtype=..., cuda: bool = ...) -> None: ...

    @staticmethod
    def sparse_tensor_constructor(
        size, dtype, sparse_dim, nnz, is_coalesced): ...
