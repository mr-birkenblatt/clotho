from typing import Callable, Dict, Tuple, Union

from _typeshed import Incomplete
from torch.distributed.elastic.multiprocessing.api import (
    MultiprocessContext as MultiprocessContext,
)
from torch.distributed.elastic.multiprocessing.api import PContext as PContext
from torch.distributed.elastic.multiprocessing.api import (
    ProcessFailure as ProcessFailure,
)
from torch.distributed.elastic.multiprocessing.api import (
    RunProcsResult as RunProcsResult,
)
from torch.distributed.elastic.multiprocessing.api import (
    SignalException as SignalException,
)
from torch.distributed.elastic.multiprocessing.api import Std as Std
from torch.distributed.elastic.multiprocessing.api import (
    SubprocessContext as SubprocessContext,
)
from torch.distributed.elastic.multiprocessing.api import to_map as to_map
from torch.distributed.elastic.utils.logging import get_logger as get_logger


log: Incomplete

def start_processes(name: str, entrypoint: Union[Callable, str], args: Dict[int, Tuple], envs: Dict[int, Dict[str, str]], log_dir: str, start_method: str = ..., redirects: Union[Std, Dict[int, Std]] = ..., tee: Union[Std, Dict[int, Std]] = ...) -> PContext: ...
