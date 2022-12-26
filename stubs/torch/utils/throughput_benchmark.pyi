from _typeshed import Incomplete


def format_time(time_us: Incomplete | None = ..., time_ms: Incomplete | None = ..., time_s: Incomplete | None = ...): ...

class ExecutionStats:
    benchmark_config: Incomplete
    def __init__(self, c_stats, benchmark_config) -> None: ...
    @property
    def latency_avg_ms(self): ...
    @property
    def num_iters(self): ...
    @property
    def iters_per_second(self): ...
    @property
    def total_time_seconds(self): ...

class ThroughputBenchmark:
    def __init__(self, module) -> None: ...
    def run_once(self, *args, **kwargs): ...
    def add_input(self, *args, **kwargs) -> None: ...
    def benchmark(self, num_calling_threads: int = ..., num_warmup_iters: int = ..., num_iters: int = ..., profiler_output_path: str = ...): ...