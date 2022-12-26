from _typeshed import Incomplete

from ._crash_handler import disable_minidumps as disable_minidumps
from ._crash_handler import enable_minidumps as enable_minidumps
from ._crash_handler import (
    enable_minidumps_on_exceptions as enable_minidumps_on_exceptions,
)
from .throughput_benchmark import ThroughputBenchmark as ThroughputBenchmark


def set_module(obj, mod) -> None: ...

cmake_prefix_path: Incomplete
