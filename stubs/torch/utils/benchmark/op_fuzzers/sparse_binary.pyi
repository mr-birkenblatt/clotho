from torch.utils.benchmark import FuzzedParameter as FuzzedParameter
from torch.utils.benchmark import FuzzedSparseTensor as FuzzedSparseTensor
from torch.utils.benchmark import Fuzzer as Fuzzer
from torch.utils.benchmark import ParameterAlias as ParameterAlias


class BinaryOpSparseFuzzer(Fuzzer):
    def __init__(self, seed, dtype=..., cuda: bool = ...) -> None: ...
