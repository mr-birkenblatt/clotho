from torch.utils import benchmark


class SpectralOpFuzzer(benchmark.Fuzzer):
    def __init__(self, *, seed: int, dtype=..., cuda: bool = ..., probability_regular: float = ...) -> None: ...
