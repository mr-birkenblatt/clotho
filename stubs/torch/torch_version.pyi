from _typeshed import Incomplete


class _LazyImport:
    def __init__(self, cls_name: str) -> None: ...
    def get_cls(self): ...
    def __call__(self, *args, **kwargs): ...
    def __instancecheck__(self, obj): ...

Version: Incomplete
InvalidVersion: Incomplete

class TorchVersion(str): ...