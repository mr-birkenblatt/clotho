from numbers import Number
from typing import Union

import torch
from _typeshed import Incomplete
from torch._six import nan as nan
from torch.distributions import constraints as constraints
from torch.distributions.exp_family import (
    ExponentialFamily as ExponentialFamily,
)
from torch.distributions.utils import lazy_property as lazy_property


class Wishart(ExponentialFamily):
    arg_constraints: Incomplete
    support: Incomplete
    has_rsample: bool
    df: Incomplete
    def __init__(self, df: Union[torch.Tensor, Number], covariance_matrix: torch.Tensor = ..., precision_matrix: torch.Tensor = ..., scale_tril: torch.Tensor = ..., validate_args: Incomplete | None = ...) -> None: ...
    def expand(self, batch_shape, _instance: Incomplete | None = ...): ...
    def scale_tril(self): ...
    def covariance_matrix(self): ...
    def precision_matrix(self): ...
    @property
    def mean(self): ...
    @property
    def mode(self): ...
    @property
    def variance(self): ...
    def rsample(self, sample_shape=..., max_try_correction: Incomplete | None = ...): ...
    def log_prob(self, value): ...
    def entropy(self): ...