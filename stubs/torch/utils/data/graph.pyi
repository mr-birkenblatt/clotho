# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors
from typing import Union

from torch.utils.data import IterDataPipe, MapDataPipe


DataPipe = Union[IterDataPipe, MapDataPipe]


def traverse(datapipe, only_datapipe: bool = ...): ...
