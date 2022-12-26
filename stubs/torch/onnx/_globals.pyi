from _typeshed import Incomplete


class _InternalGlobals:
    operator_export_type: Incomplete
    training_mode: Incomplete
    onnx_shape_inference: bool
    def __init__(self) -> None: ...
    @property
    def export_onnx_opset_version(self): ...
    @export_onnx_opset_version.setter
    def export_onnx_opset_version(self, value: int): ...

GLOBALS: Incomplete
