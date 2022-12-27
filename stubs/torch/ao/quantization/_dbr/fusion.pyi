# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


from typing import List

import torch

from .function_fusion import pattern_is_match as pattern_is_match
from .mappings import (
    known_module_fusion_patterns as known_module_fusion_patterns,
)
from .utils import get_users_of_seen_q_op_info as get_users_of_seen_q_op_info


def get_module_fusion_fqns(module: torch.nn.Module) -> List[List[str]]: ...
