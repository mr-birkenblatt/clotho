# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access,no-name-in-module,undefined-variable


from typing import NamedTuple

from _typeshed import Incomplete


class _ConvMetadata(NamedTuple):
    root: Incomplete
    transpose: Incomplete
    bn: Incomplete
    reference: Incomplete
    transpose_reference: Incomplete
    fused_conv_relu: Incomplete
    fused_conv_bn: Incomplete
    fused_conv_bn_relu: Incomplete
    qat: Incomplete
    relu_qat: Incomplete
    bn_qat: Incomplete
    bn_relu_qat: Incomplete
    func: Incomplete


def get_native_backend_config_dict(): ...
