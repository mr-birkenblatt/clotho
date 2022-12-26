from _typeshed import Incomplete
from torch._C import AnyType as AnyType
from torch._C import ComplexType as ComplexType
from torch._C import DictType as DictType
from torch._C import FloatType as FloatType
from torch._C import IntType as IntType
from torch._C import ListType as ListType
from torch._C import StringType as StringType
from torch._C import TensorType as TensorType
from torch._C import TupleType as TupleType

from .._jit_internal import Any as Any
from .._jit_internal import BroadcastingList1 as BroadcastingList1
from .._jit_internal import BroadcastingList2 as BroadcastingList2
from .._jit_internal import BroadcastingList3 as BroadcastingList3
from .._jit_internal import Dict as Dict
from .._jit_internal import is_dict as is_dict
from .._jit_internal import is_list as is_list
from .._jit_internal import is_optional as is_optional
from .._jit_internal import is_tuple as is_tuple
from .._jit_internal import is_union as is_union
from .._jit_internal import List as List
from .._jit_internal import Tuple as Tuple


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
