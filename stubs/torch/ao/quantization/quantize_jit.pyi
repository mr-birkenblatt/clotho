# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


from _typeshed import Incomplete
from torch.ao.quantization.qconfig import QConfig as QConfig
from torch.ao.quantization.quant_type import QuantType as QuantType
from torch.jit._recursive import wrap_cpp_module as wrap_cpp_module


def script_qconfig(qconfig): ...


def script_qconfig_dict(qconfig_dict): ...


def fuse_conv_bn_jit(model, inplace: bool = ...): ...


def prepare_jit(model, qconfig_dict, inplace: bool = ...): ...


def prepare_dynamic_jit(model, qconfig_dict, inplace: bool = ...): ...


def convert_jit(
    model, inplace: bool = ..., debug: bool = ...,
    preserved_attrs: Incomplete | None = ...): ...


def convert_dynamic_jit(
    model, inplace: bool = ..., debug: bool = ...,
    preserved_attrs: Incomplete | None = ...): ...


def quantize_jit(
    model, qconfig_dict, run_fn, run_args, inplace: bool = ...,
    debug: bool = ...): ...


def quantize_dynamic_jit(
    model, qconfig_dict, inplace: bool = ..., debug: bool = ...): ...
