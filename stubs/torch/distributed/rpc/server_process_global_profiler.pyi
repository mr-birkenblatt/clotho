from _typeshed import Incomplete
from torch.autograd.profiler_legacy import profile as profile


class _server_process_global_profile(profile):
    def __init__(self, *args, **kwargs) -> None: ...
    entered: bool
    def __enter__(self): ...
    function_events: Incomplete
    process_global_function_events: Incomplete
    def __exit__(self, exc_type, exc_val, exc_tb): ...
