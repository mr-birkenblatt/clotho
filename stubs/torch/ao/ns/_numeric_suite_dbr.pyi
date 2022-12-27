# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


from typing import Any, Tuple

import torch
from torch.ao.quantization._dbr.quantization_state import (
    AutoQuantizationState as AutoQuantizationState,
)


def add_loggers(
    name_a: str, model_a: torch.nn.Module, name_b: str,
        model_b: torch.nn.Module) -> Tuple[torch.nn.Module,
        torch.nn.Module]: ...


def extract_logger_info(
    model_a: torch.nn.Module, model_b: torch.nn.Module,
        model_name_to_use_for_layer_names: str) -> Any: ...
