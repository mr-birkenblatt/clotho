# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


from typing import Any, Callable

import torch
from _typeshed import Incomplete


class saved_tensors_hooks:
    pack_hook: Incomplete
    unpack_hook: Incomplete

    def __init__(
        self, pack_hook: Callable[[torch.Tensor], Any],
        unpack_hook: Callable[[Any], torch.Tensor]) -> None: ...

    def __enter__(self) -> None: ...
    def __exit__(self, *args: Any): ...


class save_on_cpu(saved_tensors_hooks):
    def __init__(self, pin_memory: bool = ...): ...
