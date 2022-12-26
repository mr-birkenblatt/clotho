import types

from _typeshed import Incomplete


def disable_global_flags() -> None: ...
def flags_frozen(): ...

class ContextProp:
    getter: Incomplete
    setter: Incomplete
    def __init__(self, getter, setter) -> None: ...
    def __get__(self, obj, objtype): ...
    def __set__(self, obj, val) -> None: ...

class PropModule(types.ModuleType):
    m: Incomplete
    def __init__(self, m, name) -> None: ...
    def __getattr__(self, attr): ...