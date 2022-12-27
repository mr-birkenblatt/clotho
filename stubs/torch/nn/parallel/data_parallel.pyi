# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


from _typeshed import Incomplete

from ..modules import Module as Module
from .parallel_apply import parallel_apply as parallel_apply
from .replicate import replicate as replicate
from .scatter_gather import gather as gather
from .scatter_gather import scatter_kwargs as scatter_kwargs


class DataParallel(Module):
    module: Incomplete
    device_ids: Incomplete
    dim: Incomplete
    output_device: Incomplete
    src_device_obj: Incomplete

    def __init__(
        self, module, device_ids: Incomplete | None = ...,
        output_device: Incomplete | None = ..., dim: int = ...) -> None: ...

    def forward(self, *inputs, **kwargs): ...
    def replicate(self, module, device_ids): ...
    def scatter(self, inputs, kwargs, device_ids): ...
    def parallel_apply(self, replicas, inputs, kwargs): ...
    def gather(self, outputs, output_device): ...


def data_parallel(
    module, inputs, device_ids: Incomplete | None = ...,
        output_device: Incomplete | None = ..., dim: int = ...,
        module_kwargs: Incomplete | None = ...): ...
