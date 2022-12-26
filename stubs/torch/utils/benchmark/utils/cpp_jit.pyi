from typing import List, Optional

from _typeshed import Incomplete
from torch.utils import cpp_extension as cpp_extension
from torch.utils.benchmark.utils._stubs import (
    CallgrindModuleType as CallgrindModuleType,
)
from torch.utils.benchmark.utils._stubs import (
    TimeitModuleType as TimeitModuleType,
)


LOCK: Incomplete
SOURCE_ROOT: Incomplete
CXX_FLAGS: Optional[List[str]]
EXTRA_INCLUDE_PATHS: List[str]
CONDA_PREFIX: Incomplete
COMPAT_CALLGRIND_BINDINGS: Optional[CallgrindModuleType]

def get_compat_bindings() -> CallgrindModuleType: ...
def compile_timeit_template(*, stmt: str, setup: str, global_setup: str) -> TimeitModuleType: ...
def compile_callgrind_template(*, stmt: str, setup: str, global_setup: str) -> str: ...
