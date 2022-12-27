# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors
from torch.utils.data.datapipes.datapipe import DataChunk


class DataChunkDF(DataChunk):
    def __iter__(self): ...
    def __len__(self): ...
