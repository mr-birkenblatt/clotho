# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


from _typeshed import Incomplete
from torch import fx as fx
from torch._lazy import computation as computation
from torch._lazy.tensor_factory_functions import as, tensor_factory_functions


        tensor_factory_functions
from typing import Any, Callable, Dict, List


debug: Incomplete


class GraphInputMatcher:
    tensor_id_to_arg_idx: Dict[int, int]
    graph_input_tensor_ids: List[int]
    graph_input_ivalues: List[Any]
    def __call__(self, args): ...

    def __init__(
        self, tensor_id_to_arg_idx, graph_input_tensor_ids,
        graph_input_ivalues) -> None: ...


class ReturnValueHandler:
    index: Incomplete
    total_count: Incomplete
    def __init__(self, lazy_out_list) -> None: ...
    def duplicate_eager_tensors(self, eager_tensor_list): ...


def force_lazy_device(model: fx.GraphModule): ...


def get_fallback_ops(): ...


def extract_compiled_graph(
    model: fx.GraphModule, example_inputs) -> Callable: ...
