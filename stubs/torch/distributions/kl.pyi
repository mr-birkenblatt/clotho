# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


from _typeshed import Incomplete
from torch._six import inf as inf

from .bernoulli import Bernoulli as Bernoulli
from .beta import Beta as Beta
from .binomial import Binomial as Binomial
from .categorical import Categorical as Categorical
from .cauchy import Cauchy as Cauchy
from .continuous_bernoulli import ContinuousBernoulli as ContinuousBernoulli
from .dirichlet import Dirichlet as Dirichlet
from .distribution import Distribution as Distribution
from .exp_family import ExponentialFamily as ExponentialFamily
from .exponential import Exponential as Exponential
from .gamma import Gamma as Gamma
from .geometric import Geometric as Geometric
from .gumbel import Gumbel as Gumbel
from .half_normal import HalfNormal as HalfNormal
from .independent import Independent as Independent
from .laplace import Laplace as Laplace
from .lowrank_multivariate_normal import (
    LowRankMultivariateNormal as LowRankMultivariateNormal,
)
from .multivariate_normal import MultivariateNormal as MultivariateNormal
from .normal import Normal as Normal
from .one_hot_categorical import OneHotCategorical as OneHotCategorical
from .pareto import Pareto as Pareto
from .poisson import Poisson as Poisson
from .transformed_distribution import (
    TransformedDistribution as TransformedDistribution,
)
from .uniform import Uniform as Uniform


def register_kl(type_p, type_q): ...


class _Match:
    types: Incomplete
    def __init__(self, *types) -> None: ...
    def __eq__(self, other): ...
    def __le__(self, other): ...


def kl_divergence(p, q): ...
