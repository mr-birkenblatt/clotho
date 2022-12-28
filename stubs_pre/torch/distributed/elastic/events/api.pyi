# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access,no-name-in-module,undefined-variable


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

    def __init__(
        self, name, run_id, message, hostname, pid, node_state,
        master_endpoint, rank, local_id, error_trace) -> None: ...
