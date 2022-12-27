# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


from _typeshed import Incomplete
from torch.cuda import nccl as nccl


def broadcast(
    tensor, devices: Incomplete | None = ..., *,
        out: Incomplete | None = ...): ...


def broadcast_coalesced(tensors, devices, buffer_size: int = ...): ...


def reduce_add(inputs, destination: Incomplete | None = ...): ...


def reduce_add_coalesced(
    inputs, destination: Incomplete | None = ..., buffer_size: int = ...): ...


def scatter(
    tensor, devices: Incomplete | None = ...,
        chunk_sizes: Incomplete | None = ..., dim: int = ...,
        streams: Incomplete | None = ..., *, out: Incomplete | None = ...): ...


def gather(
    tensors, dim: int = ..., destination: Incomplete | None = ..., *,
        out: Incomplete | None = ...): ...
