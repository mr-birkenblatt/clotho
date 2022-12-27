# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


from _typeshed import Incomplete

from ._crash_handler import disable_minidumps as disable_minidumps
from ._crash_handler import enable_minidumps as enable_minidumps
from ._crash_handler import (
    enable_minidumps_on_exceptions as enable_minidumps_on_exceptions,
)
from .throughput_benchmark import ThroughputBenchmark as ThroughputBenchmark


def set_module(obj, mod) -> None: ...


cmake_prefix_path: Incomplete
