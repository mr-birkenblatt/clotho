# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


from torch.utils._pytree import LeafSpec as LeafSpec
from torch.utils._pytree import PyTree as PyTree


        TreeSpec as TreeSpec
from typing import Any, Callable, Dict, List, Type


FlattenFuncSpec = Callable[[PyTree, TreeSpec], List]
SUPPORTED_NODES: Dict[Type[Any], Any]


def register_pytree_flatten_spec(
    typ: Any, flatten_fn_spec: FlattenFuncSpec) -> None: ...


def tree_flatten_spec(pytree: PyTree, spec: TreeSpec) -> List[Any]: ...
