import abc

import torch


class WeightedQuantizedModule(torch.nn.Module, metaclass=abc.ABCMeta):
    @classmethod
    @abc.abstractmethod
    def from_reference(cls, ref_module, output_scale, output_zero_point): ...

def hide_packed_params_repr(self, params): ...
