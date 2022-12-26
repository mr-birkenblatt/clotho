from enum import Enum

from _typeshed import Incomplete
from torch.nn.parallel import (
    DistributedDataParallel as DistributedDataParallel,
)


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

def register_ddp_comm_hook(comm_hook_type: DDPCommHookType, model: DistributedDataParallel, state: Incomplete | None = ...): ...
