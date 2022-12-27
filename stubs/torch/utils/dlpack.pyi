# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
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
