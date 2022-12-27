# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level
from _typeshed import Incomplete
from torch.utils.data.datapipes.datapipe import DFIterDataPipe, IterDataPipe


class DataFramesAsTuplesPipe(IterDataPipe):
    source_datapipe: Incomplete
    def __init__(self, source_datapipe) -> None: ...
    def __iter__(self): ...


class PerRowDataFramesPipe(DFIterDataPipe):
    source_datapipe: Incomplete
    def __init__(self, source_datapipe) -> None: ...
    def __iter__(self): ...


class ConcatDataFramesPipe(DFIterDataPipe):
    source_datapipe: Incomplete
    n_batch: Incomplete
    def __init__(self, source_datapipe, batch: int = ...) -> None: ...
    def __iter__(self): ...


class ShuffleDataFramesPipe(DFIterDataPipe):
    source_datapipe: Incomplete
    def __init__(self, source_datapipe) -> None: ...
    def __iter__(self): ...


class FilterDataFramesPipe(DFIterDataPipe):
    source_datapipe: Incomplete
    filter_fn: Incomplete
    def __init__(self, source_datapipe, filter_fn) -> None: ...
    def __iter__(self): ...


class ExampleAggregateAsDataFrames(DFIterDataPipe):
    source_datapipe: Incomplete
    columns: Incomplete
    dataframe_size: Incomplete

    def __init__(
        self,
        source_datapipe,
        dataframe_size: int = ...,
        columns: Incomplete | None = ...) -> None: ...

    def __iter__(self): ...
