from collections.abc import Iterable
from typing import Dict, Iterator, List, Optional

from _typeshed import Incomplete

from . import constants as constants
from .cuda_to_hip_mappings import CUDA_TO_HIP_MAPPINGS as CUDA_TO_HIP_MAPPINGS
from .cuda_to_hip_mappings import MATH_TRANSPILATIONS as MATH_TRANSPILATIONS


HipifyResult = Dict[str, Optional[str]]
HipifyFinalResult = Dict[str, HipifyResult]
HIPIFY_C_BREADCRUMB: str
HIPIFY_FINAL_RESULT: HipifyFinalResult
PYTORCH_TEMPLATE_MAP: Incomplete


class InputError(Exception):
    message: Incomplete
    def __init__(self, message) -> None: ...


def openf(filename, mode): ...


class bcolors:
    HEADER: str
    OKBLUE: str
    OKGREEN: str
    WARNING: str
    FAIL: str
    ENDC: str
    BOLD: str
    UNDERLINE: str


class GeneratedFileCleaner:
    keep_intermediates: Incomplete
    files_to_clean: Incomplete
    dirs_to_clean: Incomplete
    def __init__(self, keep_intermediates: bool = ...) -> None: ...
    def __enter__(self): ...
    def open(self, fn, *args, **kwargs): ...
    def makedirs(self, dn, exist_ok: bool = ...) -> None: ...
    def __exit__(self, type, value, traceback) -> None: ...


def match_extensions(filename: str, extensions: Iterable) -> bool: ...


def matched_files_iter(
    root_path: str,
    includes: Iterable = ...,
    ignores: Iterable = ...,
    extensions: Iterable = ...,
    out_of_place_only: bool = ...,
    is_pytorch_extension: bool = ...) -> Iterator[str]: ...


def preprocess_file_and_save_result(
    output_directory: str,
    filepath: str,
    all_files: Iterable,
    header_include_dirs: Iterable,
    stats: Dict[str, List],
    hip_clang_launch: bool,
    is_pytorch_extension: bool,
    clean_ctx: GeneratedFileCleaner,
    show_progress: bool) -> None: ...


def compute_stats(stats) -> None: ...


def add_dim3(kernel_string, cuda_kernel): ...


RE_KERNEL_LAUNCH: Incomplete


def processKernelLaunches(string, stats): ...


def find_closure_group(input_string, start, group): ...


def find_bracket_group(input_string, start): ...


def find_parentheses_group(input_string, start): ...


RE_ASSERT: Incomplete


def replace_math_functions(input_string): ...


RE_SYNCTHREADS: Incomplete


def hip_header_magic(input_string): ...


RE_EXTERN_SHARED: Incomplete


def replace_extern_shared(input_string): ...


def get_hip_file_path(rel_filepath, is_pytorch_extension: bool = ...): ...


def is_out_of_place(rel_filepath): ...


def is_pytorch_file(rel_filepath): ...


def is_cusparse_file(rel_filepath): ...


def is_caffe2_gpu_file(rel_filepath): ...


class Trie:
    data: Incomplete
    def __init__(self) -> None: ...
    def add(self, word) -> None: ...
    def dump(self): ...
    def quote(self, char): ...
    def pattern(self): ...


CAFFE2_TRIE: Incomplete
CAFFE2_MAP: Incomplete
PYTORCH_TRIE: Incomplete
PYTORCH_MAP: Dict[str, object]
PYTORCH_SPARSE_MAP: Incomplete
dst: Incomplete
meta_data: Incomplete
RE_CAFFE2_PREPROCESSOR: Incomplete
RE_PYTORCH_PREPROCESSOR: Incomplete
RE_QUOTE_HEADER: Incomplete
RE_ANGLE_HEADER: Incomplete
RE_THC_GENERIC_FILE: Incomplete
RE_CU_SUFFIX: Incomplete


def preprocessor(
    output_directory: str,
    filepath: str,
    all_files: Iterable,
    header_include_dirs: Iterable,
    stats: Dict[str, List],
    hip_clang_launch: bool,
    is_pytorch_extension: bool,
    clean_ctx: GeneratedFileCleaner,
    show_progress: bool) -> HipifyResult: ...


def file_specific_replacement(
    filepath, search_string, replace_string, strict: bool = ...): ...


def file_add_header(filepath, header) -> None: ...


def fix_static_global_kernels(in_txt): ...


RE_INCLUDE: Incomplete


def extract_arguments(start, string): ...


def str2bool(v): ...


def hipify(
    project_directory: str,
    show_detailed: bool = ...,
    extensions: Iterable = ...,
    header_extensions: Iterable = ...,
    output_directory: str = ...,
    header_include_dirs: Iterable = ...,
    includes: Iterable = ...,
    extra_files: Iterable = ...,
    out_of_place_only: bool = ...,
    ignores: Iterable = ...,
    show_progress: bool = ...,
    hip_clang_launch: bool = ...,
    is_pytorch_extension: bool = ...,
    hipify_extra_files_only: bool = ...,
    clean_ctx: Optional[GeneratedFileCleaner] = ...) -> HipifyFinalResult: ...
