# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level
from typing import Callable, Dict, Sequence, Union

import torch._refs
from _typeshed import Incomplete


decomposition_table: Dict[torch._ops.OpOverload, Callable]


def register_decomposition(
    aten_op, registry: Incomplete | None = ..., *,
    disable_meta: bool = ...): ...


def get_decompositions(
    aten_ops: Sequence[
        Union[torch._ops.OpOverload, torch._ops.OpOverloadPacket]],
    ) -> Dict[torch._ops.OpOverload, Callable]: ...
