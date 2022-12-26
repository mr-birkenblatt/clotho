from _typeshed import Incomplete
from torch._six import nan as nan
from torch.distributions import constraints as constraints
from torch.distributions.transformed_distribution import (
    TransformedDistribution as TransformedDistribution,
)
from torch.distributions.transforms import AffineTransform as AffineTransform
from torch.distributions.transforms import PowerTransform as PowerTransform
from torch.distributions.uniform import Uniform as Uniform
from torch.distributions.utils import broadcast_all as broadcast_all
from torch.distributions.utils import euler_constant as euler_constant


class Kumaraswamy(TransformedDistribution):
    arg_constraints: Incomplete
    support: Incomplete
    has_rsample: bool
    def __init__(self, concentration1, concentration0, validate_args: Incomplete | None = ...) -> None: ...
    def expand(self, batch_shape, _instance: Incomplete | None = ...): ...
    @property
    def mean(self): ...
    @property
    def mode(self): ...
    @property
    def variance(self): ...
    def entropy(self): ...
