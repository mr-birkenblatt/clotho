# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


import torch
from _typeshed import Incomplete


class _ParameterMeta(torch._C._TensorMeta):
    def __instancecheck__(self, instance): ...


class Parameter(torch.Tensor, metaclass=_ParameterMeta):

    def __new__(
        cls, data: Incomplete | None = ..., requires_grad: bool = ...): ...

    def __deepcopy__(self, memo): ...
    def __reduce_ex__(self, proto): ...
    __torch_function__: Incomplete


class UninitializedTensorMixin:
    data: Incomplete
    __class__: Incomplete

    def materialize(
        self, shape, device: Incomplete | None = ...,
        dtype: Incomplete | None = ...) -> None: ...

    @property
    def shape(self) -> None: ...
    def share_memory_(self) -> None: ...
    def __reduce_ex__(self, proto): ...

    @classmethod
    def __torch_function__(
        cls, func, types, args=..., kwargs: Incomplete | None = ...): ...


def is_lazy(param): ...


class UninitializedParameter(UninitializedTensorMixin, Parameter):
    cls_to_become: Incomplete

    def __new__(
        cls, requires_grad: bool = ..., device: Incomplete | None = ...,
        dtype: Incomplete | None = ...) -> None: ...


class UninitializedBuffer(UninitializedTensorMixin, torch.Tensor):
    cls_to_become: Incomplete

    def __new__(
        cls, requires_grad: bool = ..., device: Incomplete | None = ...,
        dtype: Incomplete | None = ...) -> None: ...
