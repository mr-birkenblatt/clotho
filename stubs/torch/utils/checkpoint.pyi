# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


from typing import Any, Iterable, List, Tuple

import torch


def detach_variable(inputs: Tuple[Any, ...]) -> Tuple[torch.Tensor, ...]: ...


def check_backward_validity(inputs: Iterable[Any]) -> None: ...


def get_device_states(*args) -> Tuple[List[int], List[torch.Tensor]]: ...


def set_device_states(devices, states) -> None: ...


class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, preserve_rng_state, *args): ...
    @staticmethod
    def backward(ctx, *args): ...


def checkpoint(function, *args, use_reentrant: bool = ..., **kwargs): ...


def checkpoint_sequential(functions, segments, input, **kwargs): ...
