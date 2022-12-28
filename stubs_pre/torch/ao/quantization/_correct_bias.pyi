# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access,no-name-in-module,undefined-variable


import torch.ao.ns._numeric_suite as ns
from _typeshed import Incomplete


def get_module(model, name): ...


def parent_child_names(name): ...


def get_param(module, attr): ...


class MeanShadowLogger(ns.Logger):
    count: int
    float_sum: Incomplete
    quant_sum: Incomplete
    def __init__(self) -> None: ...
    def forward(self, x, y) -> None: ...
    def clear(self) -> None: ...


def bias_correction(
    float_model, quantized_model, img_data, target_modules=...,
    neval_batches: Incomplete | None = ...) -> None: ...
