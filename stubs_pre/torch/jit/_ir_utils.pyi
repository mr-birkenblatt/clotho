# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access,no-name-in-module,undefined-variable


from typing import Union

import torch
from _typeshed import Incomplete


class _InsertPoint:
    insert_point: Incomplete
    g: Incomplete
    guard: Incomplete

    def __init__(
        self, insert_point_graph: torch._C.Graph,
        insert_point: Union[torch._C.Node, torch._C.Block]) -> None: ...

    prev_insert_point: Incomplete
    def __enter__(self) -> None: ...
    def __exit__(self, *args) -> None: ...


def insert_point_guard(
    self, insert_point: Union[torch._C.Node, torch._C.Block]): ...
