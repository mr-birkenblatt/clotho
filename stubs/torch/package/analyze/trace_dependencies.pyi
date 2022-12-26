from typing import Any, Callable, Iterable, List, Tuple


def trace_dependencies(callable: Callable[[Any], Any], inputs: Iterable[Tuple[Any, ...]]) -> List[str]: ...
