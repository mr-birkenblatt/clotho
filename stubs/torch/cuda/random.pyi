# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


from typing import Iterable, List, Union

import torch

from .. import Tensor


def get_rng_state(device: Union[int, str, torch.device] = ...) -> Tensor: ...


def get_rng_state_all() -> List[Tensor]: ...


def set_rng_state(
    new_state: Tensor, device: Union[int, str,
    torch.device] = ...) -> None: ...


def set_rng_state_all(new_states: Iterable[Tensor]) -> None: ...


def manual_seed(seed: int) -> None: ...


def manual_seed_all(seed: int) -> None: ...


def seed() -> None: ...


def seed_all() -> None: ...


def initial_seed() -> int: ...
