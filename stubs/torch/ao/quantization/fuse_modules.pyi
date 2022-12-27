# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


from _typeshed import Incomplete
from torch.ao.quantization.fuser_method_mappings import as, fuse_conv_bn


        fuse_conv_bn, fuse_conv_bn_relu as fuse_conv_bn_relu,
        get_fuser_method as get_fuser_method
from torch.nn.utils.parametrize import as, type_before_parametrizations


        type_before_parametrizations


def fuse_known_modules(
    mod_list, is_qat,
    additional_fuser_method_mapping: Incomplete | None = ...): ...


def fuse_modules(
    model, modules_to_fuse, inplace: bool = ..., fuser_func=...,
    fuse_custom_config_dict: Incomplete | None = ...): ...


def fuse_modules_qat(
    model, modules_to_fuse, inplace: bool = ..., fuser_func=...,
    fuse_custom_config_dict: Incomplete | None = ...): ...
