from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    Mapping,
    Optional,
    overload,
    Set,
    Tuple,
    TypeVar,
    Union,
)

from _typeshed import Incomplete
from torch import device as device
from torch import dtype as dtype
from torch import Tensor as Tensor

from ...utils.hooks import RemovableHandle as RemovableHandle
from ..parameter import Parameter as Parameter


T = TypeVar('T', bound='Module')

class _IncompatibleKeys: ...

def register_module_forward_pre_hook(hook: Callable[..., None]) -> RemovableHandle: ...
def register_module_forward_hook(hook: Callable[..., None]) -> RemovableHandle: ...
def register_module_backward_hook(hook: Callable[[Module, _grad_t, _grad_t], Union[None, Tensor]]) -> RemovableHandle: ...
def register_module_full_backward_hook(hook: Callable[[Module, _grad_t, _grad_t], Union[None, Tensor]]) -> RemovableHandle: ...

class Module:
    dump_patches: bool
    training: bool
    def __init__(self) -> None: ...
    forward: Callable[..., Any]
    def register_buffer(self, name: str, tensor: Optional[Tensor], persistent: bool = ...) -> None: ...
    def register_parameter(self, name: str, param: Optional[Parameter]) -> None: ...
    def add_module(self, name: str, module: Optional['Module']) -> None: ...
    def register_module(self, name: str, module: Optional['Module']) -> None: ...
    def get_submodule(self, target: str) -> Module: ...
    def get_parameter(self, target: str) -> Parameter: ...
    def get_buffer(self, target: str) -> Tensor: ...
    def get_extra_state(self) -> Any: ...
    def set_extra_state(self, state: Any): ...
    def apply(self, fn: Callable[[Module], None]) -> T: ...
    def cuda(self, device: Optional[Union[int, device]] = ...) -> T: ...
    def ipu(self, device: Optional[Union[int, device]] = ...) -> T: ...
    def xpu(self, device: Optional[Union[int, device]] = ...) -> T: ...
    def cpu(self) -> T: ...
    def type(self, dst_type: Union[dtype, str]) -> T: ...
    def float(self) -> T: ...
    def double(self) -> T: ...
    def half(self) -> T: ...
    def bfloat16(self) -> T: ...
    def to_empty(self, *, device: Union[str, device]) -> T: ...
    @overload
    def to(self, device: Optional[Union[int, device]] = ..., dtype: Optional[Union[dtype, str]] = ..., non_blocking: bool = ...) -> T: ...
    @overload
    def to(self, dtype: Union[dtype, str], non_blocking: bool = ...) -> T: ...
    @overload
    def to(self, tensor: Tensor, non_blocking: bool = ...) -> T: ...
    def register_backward_hook(self, hook: Callable[[Module, _grad_t, _grad_t], Union[None, Tensor]]) -> RemovableHandle: ...
    def register_full_backward_hook(self, hook: Callable[[Module, _grad_t, _grad_t], Union[None, Tensor]]) -> RemovableHandle: ...
    def register_forward_pre_hook(self, hook: Callable[..., None]) -> RemovableHandle: ...
    def register_forward_hook(self, hook: Callable[..., None]) -> RemovableHandle: ...
    __call__: Callable[..., Any]
    def __getattr__(self, name: str) -> Union[Tensor, 'Module']: ...
    def __setattr__(self, name: str, value: Union[Tensor, 'Module']) -> None: ...
    def __delattr__(self, name) -> None: ...
    T_destination: Incomplete
    @overload
    def state_dict(self, *, destination: T_destination, prefix: str = ..., keep_vars: bool = ...) -> T_destination: ...
    @overload
    def state_dict(self, *, prefix: str = ..., keep_vars: bool = ...) -> Dict[str, Any]: ...
    def register_load_state_dict_post_hook(self, hook): ...
    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = ...): ...
    def parameters(self, recurse: bool = ...) -> Iterator[Parameter]: ...
    def named_parameters(self, prefix: str = ..., recurse: bool = ...) -> Iterator[Tuple[str, Parameter]]: ...
    def buffers(self, recurse: bool = ...) -> Iterator[Tensor]: ...
    def named_buffers(self, prefix: str = ..., recurse: bool = ...) -> Iterator[Tuple[str, Tensor]]: ...
    def children(self) -> Iterator['Module']: ...
    def named_children(self) -> Iterator[Tuple[str, 'Module']]: ...
    def modules(self) -> Iterator['Module']: ...
    def named_modules(self, memo: Optional[Set['Module']] = ..., prefix: str = ..., remove_duplicate: bool = ...): ...
    def train(self, mode: bool = ...) -> T: ...
    def eval(self) -> T: ...
    def requires_grad_(self, requires_grad: bool = ...) -> T: ...
    def zero_grad(self, set_to_none: bool = ...) -> None: ...
    def share_memory(self) -> T: ...
    def extra_repr(self) -> str: ...
    def __dir__(self): ...
