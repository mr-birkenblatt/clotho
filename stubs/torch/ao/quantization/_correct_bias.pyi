import torch.ao.ns._numeric_suite as ns
from _typeshed import Incomplete


def get_module(model, name): ...
def parent_child_names(name): ...
def get_param(module, attr): ...

class MeanShadowLogger(ns.Logger):
    count: int
    float_sum: Incomplete
    quant_sum: Incomplete
    def __init__(self) -> None: ...
    def forward(self, x, y) -> None: ...
    def clear(self) -> None: ...

def bias_correction(float_model, quantized_model, img_data, target_modules=..., neval_batches: Incomplete | None = ...) -> None: ...
