from typing import Union

import torch
from _typeshed import Incomplete


class _InsertPoint:
    insert_point: Incomplete
    g: Incomplete
    guard: Incomplete
    def __init__(self, insert_point_graph: torch._C.Graph, insert_point: Union[torch._C.Node, torch._C.Block]) -> None: ...
    prev_insert_point: Incomplete
    def __enter__(self) -> None: ...
    def __exit__(self, *args) -> None: ...

def insert_point_guard(self, insert_point: Union[torch._C.Node, torch._C.Block]): ...
