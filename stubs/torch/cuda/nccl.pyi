# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


from typing import Optional, Sequence, Union

import torch.cuda
from _typeshed import Incomplete


def all_reduce(
    inputs, outputs: Incomplete | None = ..., op=...,
        streams: Incomplete | None = ...,
        comms: Incomplete | None = ...) -> None: ...


def reduce(
    inputs: Sequence[torch.Tensor], output: Optional[Union[torch.Tensor,
                Sequence[torch.Tensor]]] = ..., root: int = ...,
        op: int = ..., streams: Optional[Sequence[torch.cuda.Stream]] = ...,
        comms: Incomplete | None = ..., *,
        outputs: Optional[Sequence[torch.Tensor]] = ...) -> None: ...


def broadcast(
    inputs: Sequence[torch.Tensor], root: int = ...,
        streams: Incomplete | None = ...,
        comms: Incomplete | None = ...) -> None: ...


def all_gather(
    inputs: Sequence[torch.Tensor], outputs: Sequence[torch.Tensor],
        streams: Incomplete | None = ...,
        comms: Incomplete | None = ...) -> None: ...


def reduce_scatter(
    inputs: Sequence[torch.Tensor], outputs: Sequence[torch.Tensor],
        op: int = ..., streams: Incomplete | None = ...,
        comms: Incomplete | None = ...) -> None: ...
