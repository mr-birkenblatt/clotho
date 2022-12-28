# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access,no-name-in-module,undefined-variable


from _typeshed import Incomplete


TensorProtoDataType: Incomplete
OperatorExportTypes: Incomplete
TrainingMode: Incomplete
ONNX_ARCHIVE_MODEL_PROTO_NAME: str
producer_name: str
producer_version: Incomplete


class ExportTypes:
    PROTOBUF_FILE: str
    ZIP_ARCHIVE: str
    COMPRESSED_ZIP_ARCHIVE: str
    DIRECTORY: str


class CheckerError(Exception):
    ...


class SymbolicContext:
    params_dict: Incomplete
    env: Incomplete
    cur_node: Incomplete
    onnx_block: Incomplete
    def __init__(self, params_dict, env, cur_node, onnx_block) -> None: ...


def export(
    model, args, f, export_params: bool = ..., verbose: bool = ...,
    training=..., input_names: Incomplete | None = ...,
    output_names: Incomplete | None = ..., operator_export_type=...,
    opset_version: Incomplete | None = ..., do_constant_folding: bool = ...,
    dynamic_axes: Incomplete | None = ...,
    keep_initializers_as_inputs: Incomplete | None = ...,
    custom_opsets: Incomplete | None = ...,
    export_modules_as_functions: bool = ...): ...


def export_to_pretty_string(*args, **kwargs) -> str: ...


def select_model_mode_for_export(model, mode): ...


def is_in_onnx_export(): ...


def register_custom_op_symbolic(
    symbolic_name, symbolic_fn, opset_version) -> None: ...


def unregister_custom_op_symbolic(symbolic_name, opset_version) -> None: ...


def is_onnx_log_enabled(): ...


def enable_log() -> None: ...


def disable_log() -> None: ...


def set_log_stream(stream_name: str = ...) -> None: ...


def log(*args) -> None: ...
