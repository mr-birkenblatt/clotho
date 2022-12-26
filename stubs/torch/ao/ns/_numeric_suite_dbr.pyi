from typing import Any, Tuple

import torch
from torch.ao.quantization._dbr.quantization_state import (
    AutoQuantizationState as AutoQuantizationState,
)


def add_loggers(name_a: str, model_a: torch.nn.Module, name_b: str, model_b: torch.nn.Module) -> Tuple[torch.nn.Module, torch.nn.Module]: ...
def extract_logger_info(model_a: torch.nn.Module, model_b: torch.nn.Module, model_name_to_use_for_layer_names: str) -> Any: ...
