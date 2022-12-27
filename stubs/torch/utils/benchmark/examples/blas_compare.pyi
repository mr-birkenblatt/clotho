# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


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
