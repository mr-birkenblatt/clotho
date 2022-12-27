# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access,no-name-in-module,undefined-variable


from typing import Generator

import torch
from _typeshed import Incomplete
from torch._C import default_generator as default_generator


def set_rng_state(new_state: torch.Tensor) -> None: ...


def get_rng_state() -> torch.Tensor: ...


def manual_seed(seed) -> torch._C.Generator: ...


def seed() -> int: ...


def initial_seed() -> int: ...


def fork_rng(
    devices: Incomplete | None = ..., enabled: bool = ...,
    _caller: str = ..., _devices_kw: str = ...) -> Generator: ...
