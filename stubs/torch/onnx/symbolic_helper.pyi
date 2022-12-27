# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


import enum
from typing import Optional, Tuple

import torch.onnx
from _typeshed import Incomplete
from torch import _C
from torch.onnx._globals import GLOBALS as GLOBALS


def parse_args(*arg_descriptors): ...


def quantized_args(
    *arg_q_descriptors: bool, scale: Optional[float] = ...,
        zero_point: Optional[int] = ...): ...


def is_caffe2_aten_fallback(): ...


def check_training_mode(op_train_mode, op_name) -> None: ...


def dequantize_helper(
    g, qtensor: _C.Value,
        qdtype:
        Optional[torch.onnx.TensorProtoDataType] = ...) -> Tuple[_C.Value,
        _C.Value, _C.Value, Optional[_C.Value]]: ...


def quantize_helper(
    g, tensor: _C.Value, scale: _C.Value, zero_point: _C.Value,
        axis: Optional[_C.Value] = ...) -> _C.Value: ...


def requantize_bias_helper(
    g, bias, input_scale, weight_scale, axis: Incomplete | None = ...): ...


def args_have_same_dtype(args): ...


cast_pytorch_to_onnx: Incomplete
scalar_name_to_pytorch: Incomplete


class ScalarType(enum.IntEnum):
    UINT8: int
    INT8: Incomplete
    SHORT: Incomplete
    INT: Incomplete
    INT64: Incomplete
    HALF: Incomplete
    FLOAT: Incomplete
    DOUBLE: Incomplete
    COMPLEX32: Incomplete
    COMPLEX64: Incomplete
    COMPLEX128: Incomplete
    BOOL: Incomplete
    QINT8: Incomplete
    QUINT8: Incomplete
    QINT32: Incomplete
    BFLOAT16: Incomplete

scalar_type_to_pytorch_type: Incomplete
pytorch_name_to_type: Incomplete
scalar_type_to_onnx: Incomplete
