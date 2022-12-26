from typing import Any, Dict, Tuple

import torch
from torch import Tensor


def functional_call(module: torch.nn.Module, parameters_and_buffers: Dict[str, Tensor], args: Tuple, kwargs: Dict[str, Any] = ...): ...
