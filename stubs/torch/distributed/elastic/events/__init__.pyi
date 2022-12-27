# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


from enum import Enum as Enum
from typing import Optional

from torch.distributed.elastic.events.handlers import (
    get_logging_handler as get_logging_handler,
)

from .api import Event as Event
from .api import EventMetadataValue as EventMetadataValue
from .api import EventSource as EventSource
from .api import NodeState as NodeState
from .api import RdzvEvent as RdzvEvent


def record(event: Event, destination: str = ...) -> None: ...


def record_rdzv_event(event: RdzvEvent) -> None: ...


def construct_and_record_rdzv_event(
    run_id: str, message: str, node_state: NodeState, name: str = ...,
        hostname: str = ..., pid: Optional[int] = ...,
        master_endpoint: str = ..., local_id: Optional[int] = ...,
        rank: Optional[int] = ...) -> None: ...
