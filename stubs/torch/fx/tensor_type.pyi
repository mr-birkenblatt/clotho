# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


from _typeshed import Incomplete
from torch.fx.experimental.unification import Var as Var

from ._compatibility import compatibility as compatibility


class TensorType:
    __origin__: Incomplete
    __args__: Incomplete
    def __init__(self, dim) -> None: ...
    def __eq__(self, other): ...
    @staticmethod
    def __class_getitem__(*args): ...


class _DynType:
    __name__: str
    def __init__(self) -> None: ...
    def __eq__(self, other): ...


Dyn: Incomplete


def is_consistent(t1, t2): ...


def is_more_precise(t1, t2): ...
