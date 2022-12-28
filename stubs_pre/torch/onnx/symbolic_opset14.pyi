# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access,no-name-in-module,undefined-variable


from _typeshed import Incomplete
from torch.onnx import symbolic_helper as symbolic_helper
from torch.onnx._globals import GLOBALS as GLOBALS


def hardswish(g, self): ...


def tril(g, self, diagonal, out: Incomplete | None = ...): ...


def triu(g, self, diagonal, out: Incomplete | None = ...): ...


def reshape(g, self, shape): ...


def batch_norm(
    g, input, weight, bias, running_mean, running_var, training, momentum,
    eps, cudnn_enabled): ...


class Quantized:
    domain: str
    @staticmethod
    def hardswish(g, x, op_scale, op_zero_point): ...
