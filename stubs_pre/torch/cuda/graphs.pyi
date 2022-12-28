# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access,no-name-in-module,undefined-variable


import torch
from _typeshed import Incomplete


def is_current_stream_capturing(): ...


def graph_pool_handle(): ...


class CUDAGraph(torch._C._CUDAGraph):
    def __new__(cls): ...
    def __init__(self) -> None: ...
    def capture_begin(self, pool: Incomplete | None = ...) -> None: ...
    def capture_end(self) -> None: ...
    def replay(self) -> None: ...
    def reset(self) -> None: ...
    def pool(self): ...


class graph:
    default_capture_stream: Incomplete
    pool: Incomplete
    capture_stream: Incomplete
    stream_ctx: Incomplete
    cuda_graph: Incomplete

    def __init__(
        self, cuda_graph, pool: Incomplete | None = ...,
        stream: Incomplete | None = ...) -> None: ...

    def __enter__(self) -> None: ...
    def __exit__(self, exc_type, exc_value, traceback) -> None: ...


def make_graphed_callables(callables, sample_args): ...
