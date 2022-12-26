from collections.abc import Generator
from typing import List, Tuple

from _typeshed import Incomplete


def optimized_execution(should_optimize) -> Generator[None, None, None]: ...
def fuser(name) -> Generator[None, None, None]: ...

last_executed_optimized_graph: Incomplete

def set_fusion_strategy(strategy: List[Tuple[str, int]]): ...
