# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


import abc
from abc import ABC, abstractmethod

from _typeshed import Incomplete


class BasePruningMethod(ABC, metaclass=abc.ABCMeta):
    def __init__(self) -> None: ...
    def __call__(self, module, inputs) -> None: ...
    @abstractmethod
    def compute_mask(self, t, default_mask): ...
    def apply_mask(self, module): ...

    @classmethod
    def apply(
        cls, module, name, *args, importance_scores: Incomplete | None = ...,
        **kwargs): ...

    def prune(
        self, t, default_mask: Incomplete | None = ...,
        importance_scores: Incomplete | None = ...): ...

    def remove(self, module) -> None: ...


class PruningContainer(BasePruningMethod):
    def __init__(self, *args) -> None: ...
    def add_pruning_method(self, method) -> None: ...
    def __len__(self): ...
    def __iter__(self): ...
    def __getitem__(self, idx): ...
    def compute_mask(self, t, default_mask): ...


class Identity(BasePruningMethod):
    PRUNING_TYPE: str
    def compute_mask(self, t, default_mask): ...
    @classmethod
    def apply(cls, module, name): ...


class RandomUnstructured(BasePruningMethod):
    PRUNING_TYPE: str
    amount: Incomplete
    def __init__(self, amount) -> None: ...
    def compute_mask(self, t, default_mask): ...
    @classmethod
    def apply(cls, module, name, amount): ...


class L1Unstructured(BasePruningMethod):
    PRUNING_TYPE: str
    amount: Incomplete
    def __init__(self, amount) -> None: ...
    def compute_mask(self, t, default_mask): ...

    @classmethod
    def apply(
        cls, module, name, amount,
        importance_scores: Incomplete | None = ...): ...


class RandomStructured(BasePruningMethod):
    PRUNING_TYPE: str
    amount: Incomplete
    dim: Incomplete
    def __init__(self, amount, dim: int = ...) -> None: ...
    def compute_mask(self, t, default_mask): ...
    @classmethod
    def apply(cls, module, name, amount, dim: int = ...): ...


class LnStructured(BasePruningMethod):
    PRUNING_TYPE: str
    amount: Incomplete
    n: Incomplete
    dim: Incomplete
    def __init__(self, amount, n, dim: int = ...) -> None: ...
    def compute_mask(self, t, default_mask): ...

    @classmethod
    def apply(
        cls, module, name, amount, n, dim,
        importance_scores: Incomplete | None = ...): ...


class CustomFromMask(BasePruningMethod):
    PRUNING_TYPE: str
    mask: Incomplete
    def __init__(self, mask) -> None: ...
    def compute_mask(self, t, default_mask): ...
    @classmethod
    def apply(cls, module, name, mask): ...


def identity(module, name): ...


def random_unstructured(module, name, amount): ...


def l1_unstructured(
    module, name, amount, importance_scores: Incomplete | None = ...): ...


def random_structured(module, name, amount, dim): ...


def ln_structured(
    module, name, amount, n, dim,
        importance_scores: Incomplete | None = ...): ...


def global_unstructured(
    parameters, pruning_method, importance_scores: Incomplete | None = ...,
        **kwargs) -> None: ...


def custom_from_mask(module, name, mask): ...


def remove(module, name): ...


def is_pruned(module): ...
