import torch


class ReferenceQuantizedModule(torch.nn.Module):
    def get_weight(self): ...
    def get_quantized_weight(self): ...
