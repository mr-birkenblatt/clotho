from _typeshed import Incomplete
from torch._C._distributed_autograd import backward as backward
from torch._C._distributed_autograd import (
    DistAutogradContext as DistAutogradContext,
)
from torch._C._distributed_autograd import get_gradients as get_gradients


def is_available(): ...

class context:
    autograd_context: Incomplete
    def __enter__(self): ...
    def __exit__(self, type, value, traceback) -> None: ...
