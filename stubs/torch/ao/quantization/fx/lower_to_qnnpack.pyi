from typing import Dict, Tuple

from ..qconfig import QConfigAny as QConfigAny
from .graph_module import QuantizedGraphModule as QuantizedGraphModule


def lower_to_qnnpack(model: QuantizedGraphModule, qconfig_map: Dict[str, QConfigAny], node_name_to_scope: Dict[str, Tuple[str, type]]) -> QuantizedGraphModule: ...
