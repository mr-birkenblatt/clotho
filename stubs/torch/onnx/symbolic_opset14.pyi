from _typeshed import Incomplete
from torch.onnx import symbolic_helper as symbolic_helper
from torch.onnx._globals import GLOBALS as GLOBALS


def hardswish(g, self): ...
def tril(g, self, diagonal, out: Incomplete | None = ...): ...
def triu(g, self, diagonal, out: Incomplete | None = ...): ...
def reshape(g, self, shape): ...
def batch_norm(g, input, weight, bias, running_mean, running_var, training, momentum, eps, cudnn_enabled): ...

class Quantized:
    domain: str
    @staticmethod
    def hardswish(g, x, op_scale, op_zero_point): ...
