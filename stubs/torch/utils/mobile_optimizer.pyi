# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors
from enum import Enum
from typing import AnyStr, List, Optional, Set

import torch
from torch._C import MobileOptimizerType as MobileOptimizerType


class LintCode(Enum):
    BUNDLED_INPUT: int
    REQUIRES_GRAD: int
    DROPOUT: int
    BATCHNORM: int


def optimize_for_mobile(
    script_module: torch.jit.ScriptModule,
    optimization_blocklist: Optional[Set[MobileOptimizerType]] = ...,
    preserved_methods: Optional[List[AnyStr]] = ...,
    backend: str = ...) -> torch.jit.RecursiveScriptModule: ...


def generate_mobile_module_lints(script_module: torch.jit.ScriptModule): ...
