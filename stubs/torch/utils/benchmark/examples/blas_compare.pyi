from _typeshed import Incomplete

from . import blas_compare_setup as blas_compare_setup


MIN_RUN_TIME: int
NUM_REPLICATES: int
NUM_THREAD_SETTINGS: Incomplete
RESULT_FILE: Incomplete
SCRATCH_DIR: Incomplete
BLAS_CONFIGS: Incomplete

def clear_worker_pool() -> None: ...
def fill_core_pool(n: int): ...
def run_subprocess(args) -> None: ...
def main() -> None: ...
