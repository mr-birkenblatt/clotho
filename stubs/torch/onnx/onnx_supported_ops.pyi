# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


from typing import Union

from _typeshed import Incomplete
from torch import _C
from torch.onnx import symbolic_registry as symbolic_registry


class _TorchSchema:
    name: Incomplete
    overload_name: Incomplete
    arguments: Incomplete
    optional_arguments: Incomplete
    returns: Incomplete
    opsets: Incomplete
    def __init__(self, schema: Union[_C.FunctionSchema, str]) -> None: ...
    def __hash__(self): ...
    def __eq__(self, other) -> bool: ...
    def is_aten(self) -> bool: ...
    def is_backward(self) -> bool: ...


def onnx_supported_ops(): ...
