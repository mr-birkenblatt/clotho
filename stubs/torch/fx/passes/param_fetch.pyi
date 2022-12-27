# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access,no-name-in-module,undefined-variable


from typing import Any, Callable, Dict, List, Tuple, Type

import torch.nn as nn
from torch.fx._compatibility import compatibility as compatibility
from torch.fx.graph_module import GraphModule as GraphModule


def default_matching(name: str, target_version: int) -> str: ...


module_fetch_book: Dict[Type, Tuple[int, List[str], Callable[[str, int], str]]]


def extract_attrs_for_lowering(mod: nn.Module) -> Dict[str, Any]: ...


def lift_lowering_attrs_to_nodes(fx_module: GraphModule) -> None: ...
