# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


from .._jit_internal import Any as Any


        BroadcastingList1 as BroadcastingList1,
        BroadcastingList2 as BroadcastingList2,
        BroadcastingList3 as BroadcastingList3, Dict as Dict, List as List,
        Tuple as Tuple, is_dict as is_dict, is_list as is_list,
        is_optional as is_optional, is_tuple as is_tuple, is_union as is_union
from _typeshed import Incomplete
from torch._C import AnyType as AnyType
from torch._C import ComplexType as ComplexType


        DictType as DictType, FloatType as FloatType, IntType as IntType,
        ListType as ListType, StringType as StringType,
        TensorType as TensorType, TupleType as TupleType


class Module:
    name: Incomplete
    members: Incomplete
    def __init__(self, name, members) -> None: ...
    def __getattr__(self, name): ...


class EvalEnv:
    env: Incomplete
    rcb: Incomplete
    def __init__(self, rcb) -> None: ...
    def __getitem__(self, name): ...


def get_signature(fn, rcb, loc, is_method): ...


def get_param_names(fn, n_args): ...


def check_fn(fn, loc) -> None: ...


def parse_type_line(type_line, rcb, loc): ...


def get_type_line(source): ...


def split_type_line(type_line): ...


def try_real_annotations(fn, loc): ...


def try_ann_to_type(ann, loc): ...


def ann_to_type(ann, loc): ...
