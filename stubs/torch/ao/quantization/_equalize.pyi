# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


def set_module_weight(module, weight) -> None: ...


def set_module_bias(module, bias) -> None: ...


def get_module_weight(module): ...


def get_module_bias(module): ...


def max_over_ndim(input, axis_list, keepdim: bool = ...): ...


def min_over_ndim(input, axis_list, keepdim: bool = ...): ...


def channel_range(input, axis: int = ...): ...


def cross_layer_equalization(
    module1, module2, output_axis: int = ...,
        input_axis: int = ...) -> None: ...


def equalize(
    model, paired_modules_list, threshold: float = ...,
        inplace: bool = ...): ...


def converged(curr_modules, prev_modules, threshold: float = ...): ...
