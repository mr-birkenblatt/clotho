# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access,no-name-in-module,undefined-variable


from enum import Enum

import torch
from _typeshed import Incomplete
from torch.autograd.graph import save_on_cpu as save_on_cpu
from torch.utils.checkpoint import checkpoint as checkpoint


class CheckpointImpl(Enum):
    REENTRANT: Incomplete
    NO_REENTRANT: Incomplete


class CheckpointWrapper(torch.nn.Module):
    mod: Incomplete
    checkpoint_impl: Incomplete
    offload_to_cpu: Incomplete

    def __init__(
        self, mod: torch.nn.Module, checkpoint_impl: CheckpointImpl = ...,
        offload_to_cpu: bool = ...) -> None: ...

    def forward(self, *args, **kwargs): ...


def checkpoint_wrapper(
    module: torch.nn.Module, checkpoint_impl: CheckpointImpl = ...,
    offload_to_cpu: bool = ...) -> torch.nn.Module: ...
