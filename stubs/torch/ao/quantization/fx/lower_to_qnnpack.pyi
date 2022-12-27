# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


from typing import Dict, Tuple

from ..qconfig import QConfigAny as QConfigAny
from .graph_module import QuantizedGraphModule as QuantizedGraphModule


def lower_to_qnnpack(
    model: QuantizedGraphModule, qconfig_map: Dict[str, QConfigAny],
    node_name_to_scope: Dict[str, Tuple[str,
    type]]) -> QuantizedGraphModule: ...
