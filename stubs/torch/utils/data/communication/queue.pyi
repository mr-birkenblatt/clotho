from _typeshed import Incomplete


class LocalQueue:
    ops: int
    stored: int
    uid: int
    empty: int
    items: Incomplete
    name: Incomplete
    def __init__(self, name: str = ...) -> None: ...
    def put(self, item, block: bool = ...) -> None: ...
    def get(self, block: bool = ..., timeout: int = ...): ...

class ThreadingQueue:
    lock: Incomplete
    items: Incomplete
    name: Incomplete
    def __init__(self, name: str = ...) -> None: ...
    def put(self, item, block: bool = ...) -> None: ...
    def get(self, block: bool = ..., timeout: int = ...): ...
