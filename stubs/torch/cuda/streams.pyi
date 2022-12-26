import torch
from _typeshed import Incomplete


class Stream(torch._C._CudaStreamBase):
    def __new__(cls, device: Incomplete | None = ..., priority: int = ..., **kwargs): ...
    def wait_event(self, event) -> None: ...
    def wait_stream(self, stream) -> None: ...
    def record_event(self, event: Incomplete | None = ...): ...
    def query(self): ...
    def synchronize(self) -> None: ...
    def __eq__(self, o): ...
    def __hash__(self): ...

class ExternalStream(Stream):
    def __new__(cls, stream_ptr, device: Incomplete | None = ..., **kwargs): ...

class Event(torch._C._CudaEventBase):
    def __new__(cls, enable_timing: bool = ..., blocking: bool = ..., interprocess: bool = ...): ...
    @classmethod
    def from_ipc_handle(cls, device, handle): ...
    def record(self, stream: Incomplete | None = ...) -> None: ...
    def wait(self, stream: Incomplete | None = ...) -> None: ...
    def query(self): ...
    def elapsed_time(self, end_event): ...
    def synchronize(self) -> None: ...
    def ipc_handle(self): ...