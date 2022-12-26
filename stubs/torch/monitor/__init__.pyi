from torch._C._monitor import *
from torch.utils.tensorboard import SummaryWriter as SummaryWriter


STAT_EVENT: str

class TensorboardEventHandler:
    def __init__(self, writer: SummaryWriter) -> None: ...
    def __call__(self, event: Event) -> None: ...
