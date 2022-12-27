# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


from typing import NamedTuple

from _typeshed import Incomplete


TORCH_AVAILABLE: bool


class SystemEnv(NamedTuple):
    torch_version: Incomplete
    is_debug_build: Incomplete
    cuda_compiled_version: Incomplete
    gcc_version: Incomplete
    clang_version: Incomplete
    cmake_version: Incomplete
    os: Incomplete
    libc_version: Incomplete
    python_version: Incomplete
    python_platform: Incomplete
    is_cuda_available: Incomplete
    cuda_runtime_version: Incomplete
    nvidia_driver_version: Incomplete
    nvidia_gpu_models: Incomplete
    cudnn_version: Incomplete
    pip_version: Incomplete
    pip_packages: Incomplete
    conda_packages: Incomplete
    hip_compiled_version: Incomplete
    hip_runtime_version: Incomplete
    miopen_runtime_version: Incomplete
    caching_allocator_config: Incomplete
    is_xnnpack_available: Incomplete


def run(command): ...


def run_and_read_all(run_lambda, command): ...


def run_and_parse_first_match(run_lambda, command, regex): ...


def run_and_return_first_line(run_lambda, command): ...


def get_conda_packages(run_lambda): ...


def get_gcc_version(run_lambda): ...


def get_clang_version(run_lambda): ...


def get_cmake_version(run_lambda): ...


def get_nvidia_driver_version(run_lambda): ...


def get_gpu_info(run_lambda): ...


def get_running_cuda_version(run_lambda): ...


def get_cudnn_version(run_lambda): ...


def get_nvidia_smi(): ...


def get_platform(): ...


def get_mac_version(run_lambda): ...


def get_windows_version(run_lambda): ...


def get_lsb_version(run_lambda): ...


def check_release_file(run_lambda): ...


def get_os(run_lambda): ...


def get_python_platform(): ...


def get_libc_version(): ...


def get_pip_packages(run_lambda): ...


def get_cachingallocator_config(): ...


def is_xnnpack_available(): ...


def get_env_info(): ...


env_info_fmt: Incomplete


def pretty_str(envinfo): ...


def get_pretty_env_info(): ...


def main() -> None: ...
