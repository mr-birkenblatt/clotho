# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access,no-name-in-module,undefined-variable


from torch.distributed import FileStore as FileStore
from torch.distributed import PrefixStore as PrefixStore
from torch.distributed import Store as Store
from torch.distributed import TCPStore as TCPStore

from .constants import default_pg_timeout as default_pg_timeout


def register_rendezvous_handler(scheme, handler) -> None: ...


def rendezvous(url: str, rank: int = ..., world_size: int = ..., **kwargs): ...
