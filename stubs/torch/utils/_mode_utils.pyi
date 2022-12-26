class MetaInitErrorInfo:
    mode_name: str
    mode_class_name: str
    def __init__(self, mode_name, mode_class_name) -> None: ...

class _ModeInfo:
    mode_name: str
    mode_class: type
    base_mode_class: type
    def mode_class_name(self): ...
    def get_mode(self) -> None: ...
    def set_mode(self, mode) -> None: ...
    def __init__(self, mode_name, mode_class, base_mode_class) -> None: ...
