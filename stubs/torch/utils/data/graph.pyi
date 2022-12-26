from typing import Union

from torch.utils.data import IterDataPipe, MapDataPipe


DataPipe = Union[IterDataPipe, MapDataPipe]

def traverse(datapipe, only_datapipe: bool = ...): ...
