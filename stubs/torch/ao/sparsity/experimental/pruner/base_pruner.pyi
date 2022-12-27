# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access,no-name-in-module,undefined-variable


import abc

from _typeshed import Incomplete
from torch import nn as nn
from torch.ao.sparsity import BaseSparsifier as BaseSparsifier
from torch.ao.sparsity import fqn_to_module as fqn_to_module
from torch.ao.sparsity import module_to_fqn as module_to_fqn
from torch.nn.modules.container import ModuleDict as ModuleDict
from torch.nn.modules.container import ModuleList as ModuleList
from torch.nn.utils import parametrize as parametrize

from .parametrization import (
    ActivationReconstruction as ActivationReconstruction,
)
from .parametrization import BiasHook as BiasHook
from .parametrization import PruningParametrization as PruningParametrization
from .parametrization import ZeroesParametrization as ZeroesParametrization


SUPPORTED_MODULES: Incomplete
NEEDS_ZEROS: Incomplete


class BasePruner(BaseSparsifier, metaclass=abc.ABCMeta):
    prune_bias: Incomplete
    def __init__(self, defaults, also_prune_bias: bool = ...) -> None: ...
    model: Incomplete
    config: Incomplete
    def prepare(self, model, config) -> None: ...
    def squash_mask(self, use_path: bool = ..., *args, **kwargs) -> None: ...
    def get_module_pruned_outputs(self, module): ...
    def step(self, use_path: bool = ...) -> None: ...
    @abc.abstractmethod
    def update_mask(self, layer, **kwargs): ...
