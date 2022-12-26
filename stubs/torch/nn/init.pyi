from _typeshed import Incomplete
from torch import Tensor as Tensor


def calculate_gain(nonlinearity, param: Incomplete | None = ...): ...
def uniform_(tensor: Tensor, a: float = ..., b: float = ...) -> Tensor: ...
def normal_(tensor: Tensor, mean: float = ..., std: float = ...) -> Tensor: ...
def trunc_normal_(tensor: Tensor, mean: float = ..., std: float = ..., a: float = ..., b: float = ...) -> Tensor: ...
def constant_(tensor: Tensor, val: float) -> Tensor: ...
def ones_(tensor: Tensor) -> Tensor: ...
def zeros_(tensor: Tensor) -> Tensor: ...
def eye_(tensor): ...
def dirac_(tensor, groups: int = ...): ...
def xavier_uniform_(tensor: Tensor, gain: float = ...) -> Tensor: ...
def xavier_normal_(tensor: Tensor, gain: float = ...) -> Tensor: ...
def kaiming_uniform_(tensor: Tensor, a: float = ..., mode: str = ..., nonlinearity: str = ...): ...
def kaiming_normal_(tensor: Tensor, a: float = ..., mode: str = ..., nonlinearity: str = ...): ...
def orthogonal_(tensor, gain: int = ...): ...
def sparse_(tensor, sparsity, std: float = ...): ...

uniform: Incomplete
normal: Incomplete
constant: Incomplete
eye: Incomplete
dirac: Incomplete
xavier_uniform: Incomplete
xavier_normal: Incomplete
kaiming_uniform: Incomplete
kaiming_normal: Incomplete
orthogonal: Incomplete
sparse: Incomplete