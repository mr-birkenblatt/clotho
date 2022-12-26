from typing import Any, Tuple, Union

from torch.fx import Node

from .utils import Pattern as Pattern


NodePattern = Union[Tuple[Node, Node], Tuple[Node, Tuple[Node, Node]], Any]
QuantizerCls = Any
