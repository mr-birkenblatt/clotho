from torch._jit_internal import Future as Future
from torch.utils import set_module as set_module


def fork(func, *args, **kwargs): ...
def wait(future): ...