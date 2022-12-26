# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors
from _typeshed import Incomplete
from torch.utils.data.datapipes.datapipe import IterDataPipe


class IterableWrapperIterDataPipe(IterDataPipe):
    iterable: Incomplete
    deepcopy: Incomplete
    def __init__(self, iterable, deepcopy: bool = ...) -> None: ...
    def __iter__(self): ...
    def __len__(self): ...
