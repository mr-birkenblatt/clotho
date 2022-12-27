# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


from _typeshed import Incomplete
from torch.nn import Module as Module
from torch.optim.lr_scheduler import _LRScheduler


class AveragedModel(Module):
    module: Incomplete
    avg_fn: Incomplete
    use_buffers: Incomplete

    def __init__(
        self, model, device: Incomplete | None = ...,
        avg_fn: Incomplete | None = ..., use_buffers: bool = ...): ...

    def forward(self, *args, **kwargs): ...
    def update_parameters(self, model) -> None: ...


def update_bn(loader, model, device: Incomplete | None = ...) -> None: ...


class SWALR(_LRScheduler):
    anneal_func: Incomplete
    anneal_epochs: Incomplete

    def __init__(
        self, optimizer, swa_lr, anneal_epochs: int = ...,
        anneal_strategy: str = ..., last_epoch: int = ...) -> None: ...

    def get_lr(self): ...
