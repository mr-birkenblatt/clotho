from _typeshed import Incomplete


def mark_step(device: str = ..., wait: bool = ...): ...
def wait_device_ops(devices: Incomplete | None = ...) -> None: ...
def sync_multi(tensors, devices) -> None: ...
def get_tensor_id(tensor): ...
