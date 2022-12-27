# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


from typing import Callable, Dict, Optional, Tuple, Union

import torch.nn as nn
from _typeshed import Incomplete
from torch.ao.quantization.utils import get_combined_dict as get_combined_dict
from torch.ao.quantization.utils import MatchAllNode as MatchAllNode
from torch.ao.quantization.utils import Pattern as Pattern


def fuse_conv_bn(is_qat, conv, bn): ...


def fuse_conv_bn_relu(is_qat, conv, bn, relu): ...


def fuse_linear_bn(is_qat, linear, bn): ...


def fuse_convtranspose_bn(is_qat, convt, bn): ...


def sequential_wrapper2(sequential): ...


DEFAULT_OP_LIST_TO_FUSER_METHOD: Dict[Tuple, Union[nn.Sequential, Callable]]


def get_fuser_method(
    op_list, additional_fuser_method_mapping: Incomplete | None = ...): ...


def reverse_sequential_wrapper2(sequential): ...


def reverse2(f): ...


def reverse3(f): ...


DEFAULT_PATTERN_TO_FUSER_METHOD: Dict[Pattern, Union[nn.Sequential, Callable]]


def get_valid_patterns(op_pattern): ...


def get_fuser_method_new(
    op_pattern: Pattern, fuser_method_mapping: Optional[Dict[Pattern,
                    Union[nn.Sequential, Callable]]] = ...): ...
