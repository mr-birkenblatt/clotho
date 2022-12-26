import enum
from typing import Any

import torch
from _typeshed import Incomplete


class DLDeviceType(enum.IntEnum):
    kDLCPU: Incomplete
    kDLGPU: Incomplete
    kDLCPUPinned: Incomplete
    kDLOpenCL: Incomplete
    kDLVulkan: Incomplete
    kDLMetal: Incomplete
    kDLVPI: Incomplete
    kDLROCM: Incomplete
    kDLExtDev: Incomplete


def from_dlpack(ext_tensor: Any) -> torch.Tensor: ...
