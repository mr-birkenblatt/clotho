# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


from torch.utils.benchmark import Measurement as Measurement
from torch.utils.benchmark import Timer as Timer
from torch.utils.benchmark.op_fuzzers import unary as unary


def parse_args(): ...


def construct_stmt_and_label(pr, params): ...


def subprocess_main(args) -> None: ...


def merge(measurements): ...


def process_results(results, test_variance) -> None: ...


def construct_table(results, device_str, test_variance): ...


def row_str(rel_diff, diff_seconds, measurement): ...


def read_results(result_file: str): ...


def run(cmd, cuda_visible_devices: str = ...): ...


def test_source(envs) -> None: ...


def map_fn(args): ...


def main(args) -> None: ...
