import socket
from typing import Any, List


def get_env_variable_or_raise(env_name: str) -> str: ...
def get_socket_with_port() -> socket.socket: ...

class macros:
    local_rank: str
    @staticmethod
    def substitute(args: List[Any], local_rank: str) -> List[str]: ...
