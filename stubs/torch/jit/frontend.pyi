# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


from _typeshed import Incomplete
from torch._C._jit_tree_views import Apply as Apply
from torch._C._jit_tree_views import Assert as Assert
from torch._C._jit_tree_views import Assign as Assign
from torch._C._jit_tree_views import Attribute as Attribute
from torch._C._jit_tree_views import AugAssign as AugAssign
from torch._C._jit_tree_views import BinOp as BinOp
from torch._C._jit_tree_views import Break as Break
from torch._C._jit_tree_views import ClassDef as ClassDef
from torch._C._jit_tree_views import Const as Const
from torch._C._jit_tree_views import Continue as Continue
from torch._C._jit_tree_views import Decl as Decl
from torch._C._jit_tree_views import Def as Def
from torch._C._jit_tree_views import Delete as Delete
from torch._C._jit_tree_views import DictComp as DictComp
from torch._C._jit_tree_views import DictLiteral as DictLiteral
from torch._C._jit_tree_views import Dots as Dots
from torch._C._jit_tree_views import EmptyTypeAnnotation as EmptyTypeAnnotation
from torch._C._jit_tree_views import ExprStmt as ExprStmt
from torch._C._jit_tree_views import FalseLiteral as FalseLiteral
from torch._C._jit_tree_views import For as For
from torch._C._jit_tree_views import Ident as Ident
from torch._C._jit_tree_views import If as If
from torch._C._jit_tree_views import ListComp as ListComp
from torch._C._jit_tree_views import ListLiteral as ListLiteral
from torch._C._jit_tree_views import NoneLiteral as NoneLiteral
from torch._C._jit_tree_views import Param as Param
from torch._C._jit_tree_views import Pass as Pass
from torch._C._jit_tree_views import Property as Property
from torch._C._jit_tree_views import Raise as Raise
from torch._C._jit_tree_views import Return as Return
from torch._C._jit_tree_views import Select as Select
from torch._C._jit_tree_views import SliceExpr as SliceExpr
from torch._C._jit_tree_views import Starred as Starred
from torch._C._jit_tree_views import Stmt as Stmt
from torch._C._jit_tree_views import StringLiteral as StringLiteral
from torch._C._jit_tree_views import Subscript as Subscript
from torch._C._jit_tree_views import TernaryIf as TernaryIf
from torch._C._jit_tree_views import TrueLiteral as TrueLiteral
from torch._C._jit_tree_views import TupleLiteral as TupleLiteral
from torch._C._jit_tree_views import UnaryOp as UnaryOp
from torch._C._jit_tree_views import Var as Var
from torch._C._jit_tree_views import While as While
from torch._C._jit_tree_views import With as With
from torch._C._jit_tree_views import WithItem as WithItem
from torch._jit_internal import FunctionModifiers as FunctionModifiers
from torch._jit_internal import is_static_fn as is_static_fn
from torch._jit_internal import should_drop as should_drop
from torch._sources import (
    get_source_lines_and_file as get_source_lines_and_file,
)
from torch._sources import make_source_context as make_source_context
from torch._sources import parse_def as parse_def
from torch.jit._monkeytype_config import (
    get_qualified_name as get_qualified_name,
)
from torch.jit._monkeytype_config import monkeytype_trace as monkeytype_trace


def is_reserved_name(name): ...


pretty_node_names: Incomplete
node_start_tokens: Incomplete


class FrontendError(Exception):
    source_range: Incomplete
    msg: Incomplete
    error_report: Incomplete
    def __init__(self, source_range, msg) -> None: ...


class NotSupportedError(FrontendError):
    ...


class UnsupportedNodeError(NotSupportedError):
    def __init__(self, ctx, offending_node, reason: str = ...) -> None: ...


class FrontendTypeError(FrontendError):
    ...


def build_withitems(ctx, items): ...


def build_stmts(ctx, stmts): ...


def get_class_properties(cls, self_name): ...


def get_class_assigns(ctx, cls_ast): ...


def get_jit_class_def(cls, self_name): ...


def get_jit_def(
    fn, def_name, self_name: Incomplete | None = ...,
        is_classmethod: bool = ...): ...


def is_torch_jit_ignore_context_manager(stmt): ...


class Builder:
    def __call__(self, ctx, node): ...


def build_class_def(ctx, py_def, methods, properties, self_name, assigns): ...


def build_def(
    ctx, py_def, type_line, def_name, self_name: Incomplete | None = ...,
        pdt_arg_types: Incomplete | None = ...): ...


def build_param_list(
    ctx, py_args, self_name, pdt_arg_types: Incomplete | None = ...): ...


def build_param(
    ctx, py_arg, self_name, kwarg_only,
        pdt_arg_type: Incomplete | None = ...): ...


def build_ignore_context_manager(ctx, stmt): ...


def get_default_args(fn): ...


def get_default_args_for_class(cls): ...


class WithItemBuilder(Builder):
    @staticmethod
    def build_withitem(ctx, item): ...


class StmtBuilder(Builder):
    augassign_map: Incomplete
    @staticmethod
    def build_Expr(ctx, stmt): ...
    @staticmethod
    def build_Assign(ctx, stmt): ...
    @staticmethod
    def build_AnnAssign(ctx, stmt): ...
    @staticmethod
    def build_Delete(ctx, stmt): ...
    @staticmethod
    def build_Return(ctx, stmt): ...
    @staticmethod
    def build_Raise(ctx, stmt): ...
    @staticmethod
    def build_Assert(ctx, stmt): ...
    @staticmethod
    def build_AugAssign(ctx, stmt): ...
    @staticmethod
    def build_While(ctx, stmt): ...
    @staticmethod
    def build_For(ctx, stmt): ...
    @staticmethod
    def build_If(ctx, stmt): ...
    @staticmethod
    def build_Print(ctx, stmt): ...
    @staticmethod
    def build_Pass(ctx, stmt): ...
    @staticmethod
    def build_Break(ctx, stmt): ...
    @staticmethod
    def build_Continue(ctx, stmt): ...
    @staticmethod
    def build_With(ctx, stmt): ...


class ExprBuilder(Builder):
    binop_map: Incomplete
    unop_map: Incomplete
    boolop_map: Incomplete
    cmpop_map: Incomplete
    @staticmethod
    def build_Attribute(ctx, expr): ...
    @staticmethod
    def build_Call(ctx, expr): ...
    @staticmethod
    def build_Ellipsis(ctx, expr): ...
    @staticmethod
    def build_Name(ctx, expr): ...
    @staticmethod
    def build_NameConstant(ctx, expr): ...
    @staticmethod
    def build_BinOp(ctx, expr): ...
    @staticmethod
    def build_UnaryOp(ctx, expr): ...
    @staticmethod
    def build_BoolOp(ctx, expr): ...
    @staticmethod
    def build_IfExp(ctx, expr): ...
    @staticmethod
    def build_Compare(ctx, expr): ...
    @staticmethod
    def build_Subscript(ctx, expr): ...
    @staticmethod
    def build_List(ctx, expr): ...
    @staticmethod
    def build_Tuple(ctx, expr): ...
    @staticmethod
    def build_Dict(ctx, expr): ...
    @staticmethod
    def build_Num(ctx, expr): ...
    @staticmethod
    def build_Constant(ctx, expr): ...
    @staticmethod
    def build_Str(ctx, expr): ...
    @staticmethod
    def build_JoinedStr(ctx, expr): ...
    @staticmethod
    def build_ListComp(ctx, stmt): ...
    @staticmethod
    def build_DictComp(ctx, stmt): ...
    @staticmethod
    def build_Starred(ctx, expr): ...


build_expr: Incomplete
build_stmt: Incomplete
build_withitem: Incomplete


def find_before(ctx, pos, substr, offsets=...): ...
