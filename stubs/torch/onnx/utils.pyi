# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


from collections.abc import Generator

from _typeshed import Incomplete
from torch.onnx import symbolic_caffe2 as symbolic_caffe2
from torch.onnx import symbolic_helper as symbolic_helper
from torch.onnx import symbolic_registry as symbolic_registry
from torch.onnx._globals import GLOBALS as GLOBALS


def is_in_onnx_export(): ...


def select_model_mode_for_export(
    model, mode) -> Generator[None, None, None]: ...


def disable_apex_o2_state_dict_hook(model) -> Generator[None, None, None]: ...


def setup_onnx_logging(verbose) -> Generator[None, None, None]: ...


def exporter_context(
    model, mode, verbose) -> Generator[Incomplete, None, None]: ...


def export(
    model, args, f, export_params: bool = ..., verbose: bool = ...,
    training: Incomplete | None = ..., input_names: Incomplete | None = ...,
    output_names: Incomplete | None = ..., operator_export_type=...,
    opset_version: Incomplete | None = ..., do_constant_folding: bool = ...,
    dynamic_axes: Incomplete | None = ...,
    keep_initializers_as_inputs: Incomplete | None = ...,
    custom_opsets: Incomplete | None = ...,
    export_modules_as_functions: bool = ...) -> None: ...


def warn_on_static_input_change(input_states) -> None: ...


def unpack_quantized_tensor(value): ...


def export_to_pretty_string(
    model, args, export_params: bool = ..., verbose: bool = ...,
    training: Incomplete | None = ..., input_names: Incomplete | None = ...,
    output_names: Incomplete | None = ..., operator_export_type=...,
    export_type: Incomplete | None = ..., google_printer: bool = ...,
    opset_version: Incomplete | None = ...,
    keep_initializers_as_inputs: Incomplete | None = ...,
    custom_opsets: Incomplete | None = ..., add_node_names: bool = ...,
    do_constant_folding: bool = ...,
    dynamic_axes: Incomplete | None = ...): ...


def unconvertible_ops(
    model, args, training=..., opset_version: Incomplete | None = ...): ...


def get_ns_op_name_from_custom_op(symbolic_name): ...


def register_custom_op_symbolic(
    symbolic_name, symbolic_fn, opset_version) -> None: ...


def unregister_custom_op_symbolic(symbolic_name, opset_version) -> None: ...
