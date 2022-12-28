# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access,no-name-in-module,undefined-variable


import socket

from _typeshed import Incomplete
from torch.distributed.elastic.utils.logging import get_logger as get_logger


log: Incomplete


def create_c10d_store(
    is_server: bool, server_addr: str, server_port: int = ...,
    world_size: int = ..., timeout: float = ...,
    wait_for_workers: bool = ..., retries: int = ...): ...


def get_free_port(): ...


def get_socket_with_port() -> socket.socket: ...
