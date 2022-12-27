# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


def fuse_conv_bn_eval(conv, bn, transpose: bool = ...): ...


def fuse_conv_bn_weights(
    conv_w, conv_b, bn_rm, bn_rv, bn_eps, bn_w, bn_b,
    transpose: bool = ...): ...


def fuse_linear_bn_eval(linear, bn): ...


def fuse_linear_bn_weights(
    linear_w, linear_b, bn_rm, bn_rv, bn_eps, bn_w, bn_b): ...
