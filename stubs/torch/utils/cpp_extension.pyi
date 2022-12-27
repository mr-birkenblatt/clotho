# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


from typing import List, Optional, Tuple, Union

from _typeshed import Incomplete
from setuptools.command.build_ext import build_ext
from torch.torch_version import TorchVersion as TorchVersion

from ._cpp_extension_versioner import ExtensionVersioner as ExtensionVersioner
from .file_baton import FileBaton as FileBaton
from .hipify import hipify_python as hipify_python
from .hipify.hipify_python import GeneratedFileCleaner as GeneratedFileCleaner


IS_WINDOWS: Incomplete
IS_MACOS: Incomplete
IS_LINUX: Incomplete
LIB_EXT: Incomplete
EXEC_EXT: Incomplete
CLIB_PREFIX: Incomplete
CLIB_EXT: Incomplete
SHARED_FLAG: Incomplete
TORCH_LIB_PATH: Incomplete
BUILD_SPLIT_CUDA: Incomplete
SUBPROCESS_DECODE_ARGS: Incomplete
MINIMUM_GCC_VERSION: Incomplete
MINIMUM_MSVC_VERSION: Incomplete
CUDA_GCC_VERSIONS: Incomplete
CUDA_CLANG_VERSIONS: Incomplete
ABI_INCOMPATIBILITY_WARNING: str
WRONG_COMPILER_WARNING: str
CUDA_MISMATCH_MESSAGE: str
CUDA_MISMATCH_WARN: str
CUDA_NOT_FOUND_MESSAGE: str
ROCM_HOME: Incomplete
MIOPEN_HOME: Incomplete
HIP_HOME: Incomplete
IS_HIP_EXTENSION: Incomplete
ROCM_VERSION: Incomplete
CUDA_HOME: Incomplete
CUDNN_HOME: Incomplete
BUILT_FROM_SOURCE_VERSION_PATTERN: Incomplete
COMMON_MSVC_FLAGS: Incomplete
MSVC_IGNORE_CUDAFE_WARNINGS: Incomplete
COMMON_NVCC_FLAGS: Incomplete
COMMON_HIP_FLAGS: Incomplete
COMMON_HIPCC_FLAGS: Incomplete
JIT_EXTENSION_VERSIONER: Incomplete
PLAT_TO_VCVARS: Incomplete


def get_default_build_root() -> str: ...


def check_compiler_ok_for_platform(compiler: str) -> bool: ...


def get_compiler_abi_compatibility_and_version(
    compiler) -> Tuple[bool, TorchVersion]: ...


class BuildExtension(build_ext):
    @classmethod
    def with_options(cls, **options): ...
    no_python_abi_suffix: Incomplete
    use_ninja: Incomplete
    def __init__(self, *args, **kwargs) -> None: ...
    force: bool
    def finalize_options(self) -> None: ...
    cflags: Incomplete
    def build_extensions(self) -> None: ...
    def get_ext_filename(self, ext_name): ...


def CppExtension(name, sources, *args, **kwargs): ...


def CUDAExtension(name, sources, *args, **kwargs): ...


def include_paths(cuda: bool = ...) -> List[str]: ...


def library_paths(cuda: bool = ...) -> List[str]: ...


def load(
    name, sources: Union[str, List[str]],
    extra_cflags: Incomplete | None = ...,
    extra_cuda_cflags: Incomplete | None = ...,
    extra_ldflags: Incomplete | None = ...,
    extra_include_paths: Incomplete | None = ...,
    build_directory: Incomplete | None = ..., verbose: bool = ...,
    with_cuda: Optional[bool] = ..., is_python_module: bool = ...,
    is_standalone: bool = ..., keep_intermediates: bool = ...): ...


def load_inline(
    name, cpp_sources, cuda_sources: Incomplete | None = ...,
    functions: Incomplete | None = ...,
    extra_cflags: Incomplete | None = ...,
    extra_cuda_cflags: Incomplete | None = ...,
    extra_ldflags: Incomplete | None = ...,
    extra_include_paths: Incomplete | None = ...,
    build_directory: Incomplete | None = ..., verbose: bool = ...,
    with_cuda: Incomplete | None = ..., is_python_module: bool = ...,
    with_pytorch_error_handling: bool = ...,
    keep_intermediates: bool = ...): ...


def is_ninja_available(): ...


def verify_ninja_availability() -> None: ...
