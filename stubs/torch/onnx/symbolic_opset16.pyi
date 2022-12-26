from torch.nn.functional import (
    GRID_SAMPLE_INTERPOLATION_MODES as GRID_SAMPLE_INTERPOLATION_MODES,
)
from torch.nn.functional import (
    GRID_SAMPLE_PADDING_MODES as GRID_SAMPLE_PADDING_MODES,
)
from torch.onnx import symbolic_helper as symbolic_helper


def grid_sampler(g, input, grid, mode_enum, padding_mode_enum, align_corners): ...
