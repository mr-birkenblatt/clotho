# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access,no-name-in-module,undefined-variable


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


def start_processes(
    name: str, entrypoint: Union[Callable, str], args: Dict[int, Tuple],
    envs: Dict[int, Dict[str, str]], log_dir: str, start_method: str = ...,
    redirects: Union[Std, Dict[int, Std]] = ..., tee: Union[Std, Dict[int,
                    Std]] = ...) -> PContext: ...
