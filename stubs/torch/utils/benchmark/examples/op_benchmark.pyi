from torch.utils.benchmark import Timer as Timer
from torch.utils.benchmark.op_fuzzers.binary import (
    BinaryOpFuzzer as BinaryOpFuzzer,
)
from torch.utils.benchmark.op_fuzzers.unary import (
    UnaryOpFuzzer as UnaryOpFuzzer,
)


def assert_dicts_equal(dict_0, dict_1) -> None: ...
def run(n, stmt, fuzzer_cls): ...
def main() -> None: ...
