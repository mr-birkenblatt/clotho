# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


from collections.abc import Generator
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from _typeshed import Incomplete


class FuzzedParameter:
    strict: Incomplete

    def __init__(
        self, name: str, minval: Optional[Union[int, float]] = ...,
        maxval: Optional[Union[int, float]] = ...,
        distribution: Optional[Union[str, Dict[Any, float]]] = ...,
        strict: bool = ...) -> None: ...

    @property
    def name(self): ...
    def sample(self, state): ...


class ParameterAlias:
    alias_to: Incomplete
    def __init__(self, alias_to) -> None: ...


class FuzzedTensor:

    def __init__(
        self, name: str, size: Tuple[Union[str, int], ...],
        steps: Optional[Tuple[Union[str, int], ...]] = ...,
        probability_contiguous: float = ...,
        min_elements: Optional[int] = ..., max_elements: Optional[int] = ...,
        max_allocation_bytes: Optional[int] = ...,
        dim_parameter: Optional[str] = ...,
        roll_parameter: Optional[str] = ..., dtype=..., cuda: bool = ...,
        tensor_constructor: Optional[Callable] = ...) -> None: ...

    @property
    def name(self): ...
    @staticmethod
    def default_tensor_constructor(size, dtype, **kwargs): ...
    def satisfies_constraints(self, params): ...


class Fuzzer:

    def __init__(
        self, parameters: List[Union[FuzzedParameter,
        List[FuzzedParameter]]], tensors: List[Union[FuzzedTensor,
        List[FuzzedTensor]]], constraints: Optional[List[Callable]] = ...,
        seed: Optional[int] = ...) -> None: ...

    def take(self, n) -> Generator[Incomplete, None, None]: ...
    @property
    def rejection_rate(self): ...
