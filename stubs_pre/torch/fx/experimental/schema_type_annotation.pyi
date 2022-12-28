# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access,no-name-in-module,undefined-variable


from typing import Any, Dict, Tuple

import torch.fx
from _typeshed import Incomplete
from torch._jit_internal import boolean_dispatched as boolean_dispatched
from torch.fx import Transformer as Transformer
from torch.fx.node import Argument as Argument
from torch.fx.node import Target as Target


class AnnotateTypesWithSchema(Transformer):
    annotate_functionals: Incomplete
    annotate_modules: Incomplete
    annotate_get_attrs: Incomplete

    def __init__(
        self, module: torch.nn.Module, annotate_functionals: bool = ...,
        annotate_modules: bool = ...,
        annotate_get_attrs: bool = ...) -> None: ...

    def call_function(
        self, target: Target, args: Tuple[Argument, ...], kwargs: Dict[str,
                Any]): ...

    def call_module(
        self, target: Target, args: Tuple[Argument, ...], kwargs: Dict[str,
                Any]): ...

    def get_attr(
        self, target: torch.fx.node.Target, args: Tuple[Argument, ...],
        kwargs: Dict[str, Any]): ...
