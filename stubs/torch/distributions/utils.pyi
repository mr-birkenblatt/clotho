from _typeshed import Incomplete
from torch.overrides import is_tensor_like as is_tensor_like


euler_constant: float

def broadcast_all(*values): ...
def logits_to_probs(logits, is_binary: bool = ...): ...
def clamp_probs(probs): ...
def probs_to_logits(probs, is_binary: bool = ...): ...

class lazy_property:
    wrapped: Incomplete
    def __init__(self, wrapped) -> None: ...
    def __get__(self, instance, obj_type: Incomplete | None = ...): ...

class _lazy_property_and_property(lazy_property, property):
    def __init__(self, wrapped): ...

def tril_matrix_to_vec(mat, diag: int = ...): ...
def vec_to_tril_matrix(vec, diag: int = ...): ...
