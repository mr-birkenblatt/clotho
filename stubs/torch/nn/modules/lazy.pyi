# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


from _typeshed import Incomplete
from typing_extensions import Protocol

from ..parameter import is_lazy as is_lazy


class _LazyProtocol(Protocol):
    def register_forward_pre_hook(self, hook) -> None: ...


class LazyModuleMixin:
    cls_to_become: Incomplete
    def __init__(self, *args, **kwargs) -> None: ...
    def initialize_parameters(self, *args, **kwargs): ...
    def has_uninitialized_params(self): ...
