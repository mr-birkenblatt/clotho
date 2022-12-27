# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


from _typeshed import Incomplete
from torch.fx.experimental.refinement_types import Equality as Equality
from torch.fx.experimental.unification import Var as Var
from torch.fx.node import Node as Node
from torch.fx.node import Target as Target
from torch.fx.tensor_type import Dyn as Dyn
from torch.fx.tensor_type import TensorType as TensorType


        is_consistent as is_consistent, is_more_precise as is_more_precise
from torch.nn.modules.batchnorm import BatchNorm2d as BatchNorm2d
from torch.nn.modules.conv import Conv2d as Conv2d


HAS_SYMPY: bool


def expand_to_tensor_dim(t, n): ...


def broadcast_types(t1, t2): ...


def register_inference_rule(call_target): ...


def register_refinement_rule(call_target): ...


def register_algebraic_expressions_inference_rule(call_target): ...


def add_inference_rule(n: Node): ...


def get_attr_inference_rule(n: Node, traced): ...


def transpose_inference_rule(n: Node): ...


def reshape_inference_rule(n: Node): ...


def bn2d_inference_rule(n: Node, module_instance): ...


def calculate_out_dimension(d_in, module_instance, index): ...


def get_greatest_upper_bound(type1, type2): ...


def conv2d_inference_rule(n: Node, module_instance): ...


def relu_inference_rule(n: Node, module_instance): ...


def maxpool2d_check(typ, module_instance): ...


def maxpool2d_inference_rule(n: Node, module_instance): ...


def linear_check(tensor_type, module_instance): ...


def linear_inference_rule(n: Node, module_instance): ...


def adaptiveavgpool2d_check(tensor_type, module_instance): ...


def adaptiveavgpool2d_inference_rule(n: Node, module_instance): ...


def flatten_check(tensor_type, start_dim, end_dim): ...


def flatten_inference_rule(n: Node): ...


class GraphTypeChecker:
    env: Incomplete
    traced: Incomplete
    def __init__(self, env, traced) -> None: ...
    def type_check(self): ...
    def type_check_node(self, n: Node): ...


def conv_refinement_rule(n: Node): ...


def linear_refinement_rule(n: Node): ...


def all_eq(n: Node): ...


def first_two_eq(n: Node): ...


def element_wise_eq(n: Node): ...


def flatten_refinement_rule(n: Node): ...


def conv_rule(n: Node, module_instance): ...


class Refine:
    constraints: Incomplete
    traced: Incomplete
    symbol_iter: Incomplete
    def __init__(self, traced) -> None: ...
    def refine(self): ...
    def symbolic_relations(self): ...
    def replace_dyn_with_fresh_var(self, typ): ...
    def convert_to_sympy_symbols(self, typ): ...
    def refine_node(self, n: Node): ...
    def infer_symbolic_relations(self, n: Node): ...


def get_parameter(traced, target: str): ...
