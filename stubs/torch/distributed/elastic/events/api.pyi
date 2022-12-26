from enum import Enum
from typing import Dict, Optional, Union


EventMetadataValue = Union[str, int, float, bool, None]

class EventSource(str, Enum):
    AGENT: str
    WORKER: str

class Event:
    name: str
    source: EventSource
    timestamp: int
    metadata: Dict[str, EventMetadataValue]
    @staticmethod
    def deserialize(data: Union[str, 'Event']) -> Event: ...
    def serialize(self) -> str: ...
    def __init__(self, name, source, timestamp, metadata) -> None: ...

class NodeState(str, Enum):
    INIT: str
    RUNNING: str
    SUCCEEDED: str
    FAILED: str

class RdzvEvent:
    name: str
    run_id: str
    message: str
    hostname: str
    pid: int
    node_state: NodeState
    master_endpoint: str
    rank: Optional[int]
    local_id: Optional[int]
    error_trace: str
    @staticmethod
    def deserialize(data: Union[str, 'RdzvEvent']) -> RdzvEvent: ...
    def serialize(self) -> str: ...
    def __init__(self, name, run_id, message, hostname, pid, node_state, master_endpoint, rank, local_id, error_trace) -> None: ...
