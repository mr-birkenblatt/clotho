from typing import NamedTuple

from _typeshed import Incomplete


WORKING_ROOT: str
MKL_2020_3: str
MKL_2020_0: str
OPEN_BLAS: str
EIGEN: str
GENERIC_ENV_VARS: Incomplete
BASE_PKG_DEPS: Incomplete

class SubEnvSpec(NamedTuple):
    generic_installs: Incomplete
    special_installs: Incomplete
    environment_variables: Incomplete
    expected_blas_symbols: Incomplete
    expected_mkl_version: Incomplete
SUB_ENVS: Incomplete

def conda_run(*args): ...
def main() -> None: ...
