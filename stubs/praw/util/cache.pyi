from typing import Any, Callable, Optional

from _typeshed import Incomplete

class cachedproperty:
    func: Incomplete
    __doc__: Incomplete
    def __init__(self, func: Callable[[Any], Any], doc: Optional[str] = ...) -> None: ...
    def __call__(self, *args, **kwargs) -> None: ...
    def __get__(self, obj: Optional[Any], objtype: Optional[Any] = ...) -> Any: ...
