from typing import NamedTuple

import torch.fft
from _typeshed import Incomplete
from torch.utils import benchmark as benchmark
from torch.utils.benchmark.op_fuzzers.spectral import (
    SpectralOpFuzzer as SpectralOpFuzzer,
)


def run_benchmark(name: str, function: object, dtype: torch.dtype, seed: int, device: str, samples: int, probability_regular: float): ...

class Benchmark(NamedTuple):
    name: Incomplete
    function: Incomplete
    dtype: Incomplete
BENCHMARKS: Incomplete
BENCHMARK_MAP: Incomplete
BENCHMARK_NAMES: Incomplete
DEVICE_NAMES: Incomplete
