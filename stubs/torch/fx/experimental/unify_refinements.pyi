# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


from torch.fx.experimental.graph_gradual_typechecker import Refine as Refine
from torch.fx.experimental.unification import unify as unify
from torch.fx.experimental.unification import Var as Var
from torch.fx.tensor_type import TensorType as TensorType


def infer_symbolic_types_single_pass(traced) -> None: ...


def infer_symbolic_types(traced) -> None: ...


def convert_eq(list_of_eq): ...


def unify_eq(list_of_eq): ...


def substitute_solution_one_type(mapping, t): ...


def substitute_all_types(graph, mapping) -> None: ...


def check_for_type_equality(g1, g2): ...
