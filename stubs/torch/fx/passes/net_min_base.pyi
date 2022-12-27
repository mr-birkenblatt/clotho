# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


import torch.fx

from .shape_prop import ShapeProp as ShapeProp
from .split_utils import split_by_tags as split_by_tags
from .tools_common import CALLABLE_NODE_OPS as CALLABLE_NODE_OPS


        FxNetAccFusionsFinder as FxNetAccFusionsFinder, Names as Names,
        NodeList as NodeList, NodeSet as NodeSet,
        TensorOrTensors as TensorOrTensors, Tensors as Tensors
from typing import Callable, Optional, Tuple

from _typeshed import Incomplete
from torch.fx._compatibility import compatibility as compatibility
from torch.fx.node import map_arg as map_arg


class FxNetMinimizerBadModuleError(Exception):
    ...


class FxNetMinimizerRunFuncError(Exception):
    ...


class FxNetMinimizerResultMismatchError(Exception):
    ...


class _MinimizerSettingBase:
    accumulate_error: bool
    traverse_method: str
    find_all: bool
    return_intermediate: bool

    def __init__(
        self, accumulate_error, traverse_method, find_all,
        return_intermediate) -> None: ...


class _MinimizerBase:
    module: Incomplete
    sample_input: Incomplete
    compare_fn: Incomplete
    settings: Incomplete
    a_outputs: Incomplete
    b_outputs: Incomplete
    results: Incomplete
    fusions: Incomplete

    def __init__(
        self, module: torch.fx.GraphModule, sample_input: Tensors,
        compare_fn: Callable[[TensorOrTensors, TensorOrTensors, Names],
        Tuple[float, bool]], settings: _MinimizerSettingBase) -> None: ...

    def run_a(
        self, mod: torch.fx.GraphModule,
        inputs: Tensors) -> TensorOrTensors: ...

    def run_b(
        self, mod: torch.fx.GraphModule,
        inputs: Tensors) -> TensorOrTensors: ...

    def run_nodes(
        self, start: Optional[str] = ..., end: Optional[str] = ...): ...

    def minimize(
        self, start: Optional[str] = ...,
        end: Optional[str] = ...) -> NodeSet: ...
