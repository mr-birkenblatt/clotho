from _typeshed import Incomplete
from torch.distributed.run import get_args_parser as get_args_parser
from torch.distributed.run import run as run


logger: Incomplete

def parse_args(args): ...
def launch(args) -> None: ...
def main(args: Incomplete | None = ...) -> None: ...
