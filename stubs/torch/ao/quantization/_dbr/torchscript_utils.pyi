import torch
from torch.jit._recursive import wrap_cpp_module as wrap_cpp_module


def remove_redundant_aliases(scripted_module: torch.nn.Module): ...
