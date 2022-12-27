# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
from typing import Any

from _typeshed import Incomplete


class RemovableHandle:
    id: int
    next_id: int
    hooks_dict_ref: Incomplete
    def __init__(self, hooks_dict: Any) -> None: ...
    def remove(self) -> None: ...
    def __enter__(self) -> 'RemovableHandle': ...
    def __exit__(self, type: Any, value: Any, tb: Any) -> None: ...


def unserializable_hook(f): ...


def warn_if_has_hooks(tensor) -> None: ...


class BackwardHook:
    user_hooks: Incomplete
    module: Incomplete
    grad_outputs: Incomplete
    n_outputs: int
    output_tensors_index: Incomplete
    n_inputs: int
    input_tensors_index: Incomplete
    def __init__(self, module, user_hooks) -> None: ...
    def setup_input_hook(self, args): ...
    def setup_output_hook(self, args): ...
