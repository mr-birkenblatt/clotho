from torch.distributed import FileStore as FileStore
from torch.distributed import PrefixStore as PrefixStore
from torch.distributed import Store as Store
from torch.distributed import TCPStore as TCPStore

from .constants import default_pg_timeout as default_pg_timeout


def register_rendezvous_handler(scheme, handler) -> None: ...
def rendezvous(url: str, rank: int = ..., world_size: int = ..., **kwargs): ...
