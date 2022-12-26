import socket

from _typeshed import Incomplete
from torch.distributed.elastic.utils.logging import get_logger as get_logger


log: Incomplete

def create_c10d_store(is_server: bool, server_addr: str, server_port: int = ..., world_size: int = ..., timeout: float = ..., wait_for_workers: bool = ..., retries: int = ...): ...
def get_free_port(): ...
def get_socket_with_port() -> socket.socket: ...
