from typing import Dict, List, Optional

from _typeshed import Incomplete
from torch._C._distributed_rpc import _TensorPipeRpcBackendOptionsBase


DeviceType: Incomplete

class TensorPipeRpcBackendOptions(_TensorPipeRpcBackendOptionsBase):
    def __init__(self, *, num_worker_threads: int = ..., rpc_timeout: float = ..., init_method: str = ..., device_maps: Optional[Dict[str, Dict[DeviceType, DeviceType]]] = ..., devices: Optional[List[DeviceType]] = ..., _transports: Optional[List] = ..., _channels: Optional[List] = ...) -> None: ...
    def set_device_map(self, to: str, device_map: Dict[DeviceType, DeviceType]): ...
    devices: Incomplete
    def set_devices(self, devices: List[DeviceType]): ...
