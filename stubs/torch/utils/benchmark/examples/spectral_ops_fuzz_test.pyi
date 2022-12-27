# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


from typing import NamedTuple

import torch.fft
from _typeshed import Incomplete
from torch.utils import benchmark as benchmark
from torch.utils.benchmark.op_fuzzers.spectral import (
    SpectralOpFuzzer as SpectralOpFuzzer,
)


def run_benchmark(
    name: str, function: object, dtype: torch.dtype, seed: int, device: str,
        samples: int, probability_regular: float): ...


class Benchmark(NamedTuple):
    name: Incomplete
    function: Incomplete
    dtype: Incomplete
BENCHMARKS: Incomplete
BENCHMARK_MAP: Incomplete
BENCHMARK_NAMES: Incomplete
DEVICE_NAMES: Incomplete
