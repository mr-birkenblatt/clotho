# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors
from _typeshed import Incomplete
from torch.utils.data.datapipes.datapipe import MapDataPipe


class SequenceWrapperMapDataPipe(MapDataPipe):
    sequence: Incomplete
    def __init__(self, sequence, deepcopy: bool = ...) -> None: ...
    def __getitem__(self, index): ...
    def __len__(self): ...
