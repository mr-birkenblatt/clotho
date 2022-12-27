# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


from _typeshed import Incomplete
from torch.onnx import symbolic_helper as symbolic_helper


block_listed_operators: Incomplete
upsample_nearest1d: Incomplete
upsample_nearest2d: Incomplete
upsample_nearest3d: Incomplete
upsample_linear1d: Incomplete
upsample_bilinear2d: Incomplete
upsample_trilinear3d: Incomplete


def gt(g, input, other): ...


def lt(g, input, other): ...


def bmm(g, self, other): ...


def matmul(g, self, other): ...


def prelu(g, self, weight): ...


def mm(g, self, other): ...


def addmm(g, self, mat1, mat2, beta, alpha): ...


def flatten(g, input, start_dim, end_dim): ...


def empty(
    g, sizes, dtype, layout, device, pin_memory: bool = ...,
        memory_format: Incomplete | None = ...): ...


def empty_like(
    g, input, dtype, layout, device, pin_memory: bool = ...,
        memory_format: Incomplete | None = ...): ...


def zeros(g, sizes, dtype, layout, device, pin_memory: bool = ...): ...


def zeros_like(
    g, input, dtype, layout, device, pin_memory: bool = ...,
        memory_format: Incomplete | None = ...): ...


def ones(g, sizes, dtype, layout, device, pin_memory: bool = ...): ...


def ones_like(
    g, input, dtype, layout, device, pin_memory: bool = ...,
        memory_format: Incomplete | None = ...): ...


def full(g, sizes, value, dtype, layout, device, pin_memory: bool = ...): ...


def full_like(
    g, input, fill_value, dtype, layout, device, pin_memory: bool = ...,
        memory_format: Incomplete | None = ...): ...


def repeat(g, self, repeats): ...
