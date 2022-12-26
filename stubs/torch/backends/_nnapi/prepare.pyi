from typing import List, Optional

import torch
from _typeshed import Incomplete


class NnapiModule(torch.nn.Module):
    comp: Optional[torch.classes._nnapi.Compilation]
    weights: List[torch.Tensor]
    out_templates: List[torch.Tensor]
    shape_compute_module: Incomplete
    ser_model: Incomplete
    inp_mem_fmts: Incomplete
    out_mem_fmts: Incomplete
    def __init__(self, shape_compute_module: torch.nn.Module, ser_model: torch.Tensor, weights: List[torch.Tensor], inp_mem_fmts: List[int], out_mem_fmts: List[int]) -> None: ...
    def init(self, args: List[torch.Tensor]): ...
    def forward(self, args: List[torch.Tensor]) -> List[torch.Tensor]: ...

def convert_model_to_nnapi(model, inputs, serializer: Incomplete | None = ..., return_shapes: Incomplete | None = ..., use_int16_for_qint16: bool = ...): ...
def process_for_nnapi(model, inputs, serializer: Incomplete | None = ..., return_shapes: Incomplete | None = ..., use_int16_for_qint16: bool = ...): ...
