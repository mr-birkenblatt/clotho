# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


import abc
from typing import Dict, Optional, Tuple

from _typeshed import Incomplete
from torch import nn as nn
from torch.nn.utils import parametrize as parametrize

from .utils import FakeSparsity as FakeSparsity
from .utils import fqn_to_module as fqn_to_module
from .utils import module_to_fqn as module_to_fqn


SUPPORTED_MODULES: Incomplete


class BaseSparsifier(abc.ABC, metaclass=abc.ABCMeta):
    defaults: Incomplete
    state: Incomplete
    module_groups: Incomplete
    enable_mask_update: bool
    def __init__(self, defaults) -> None: ...
    def state_dict(self): ...
    def load_state_dict(self, state_dict, strict: bool = ...) -> None: ...
    model: Incomplete
    config: Incomplete
    def prepare(self, model, config) -> None: ...

    def squash_mask(
        self, params_to_keep: Optional[Tuple[str, ...]] = ...,
        params_to_keep_per_layer: Optional[Dict[str, Tuple[str, ...]]] = ...,
        *args, **kwargs): ...

    def convert(self) -> None: ...
    def step(self, use_path: bool = ...) -> None: ...
    @abc.abstractmethod
    def update_mask(self, layer, **kwargs): ...
