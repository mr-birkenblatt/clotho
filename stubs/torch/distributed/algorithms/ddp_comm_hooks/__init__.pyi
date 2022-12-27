# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


from enum import Enum

from _typeshed import Incomplete
from torch.nn.parallel import as, DistributedDataParallel


        DistributedDataParallel


class DDPCommHookType(Enum):
    ALLREDUCE: Incomplete
    FP16_COMPRESS: Incomplete
    BF16_COMPRESS: Incomplete
    QUANTIZE_PER_TENSOR: Incomplete
    QUANTIZE_PER_CHANNEL: Incomplete
    POWER_SGD: Incomplete
    POWER_SGD_RANK2: Incomplete
    BATCHED_POWER_SGD: Incomplete
    BATCHED_POWER_SGD_RANK2: Incomplete
    NOOP: Incomplete


def register_ddp_comm_hook(
    comm_hook_type: DDPCommHookType, model: DistributedDataParallel,
    state: Incomplete | None = ...): ...
