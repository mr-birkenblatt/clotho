from collections import OrderedDict
from typing import (
    Any,
    Iterable,
    Iterator,
    Mapping,
    Optional,
    overload,
    Tuple,
    TypeVar,
    Union,
)

from ..parameter import Parameter as Parameter
from .module import Module as Module


T = TypeVar('T', bound=Module)

class Container(Module):
    def __init__(self, **kwargs: Any) -> None: ...

class Sequential(Module):
    @overload
    def __init__(self, *args: Module) -> None: ...
    @overload
    def __init__(self, arg: OrderedDict[str, Module]) -> None: ...
    def __getitem__(self, idx) -> Union['Sequential', T]: ...
    def __setitem__(self, idx: int, module: Module) -> None: ...
    def __delitem__(self, idx: Union[slice, int]) -> None: ...
    def __len__(self) -> int: ...
    def __dir__(self): ...
    def __iter__(self) -> Iterator[Module]: ...
    def forward(self, input): ...
    def append(self, module: Module) -> Sequential: ...

class ModuleList(Module):
    def __init__(self, modules: Optional[Iterable[Module]] = ...) -> None: ...
    def __getitem__(self, idx: int) -> Union[Module, 'ModuleList']: ...
    def __setitem__(self, idx: int, module: Module) -> None: ...
    def __delitem__(self, idx: Union[int, slice]) -> None: ...
    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[Module]: ...
    def __iadd__(self, modules: Iterable[Module]) -> ModuleList: ...
    def __add__(self, other: Iterable[Module]) -> ModuleList: ...
    def __dir__(self): ...
    def insert(self, index: int, module: Module) -> None: ...
    def append(self, module: Module) -> ModuleList: ...
    def extend(self, modules: Iterable[Module]) -> ModuleList: ...

class ModuleDict(Module):
    def __init__(self, modules: Optional[Mapping[str, Module]] = ...) -> None: ...
    def __getitem__(self, key: str) -> Module: ...
    def __setitem__(self, key: str, module: Module) -> None: ...
    def __delitem__(self, key: str) -> None: ...
    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[str]: ...
    def __contains__(self, key: str) -> bool: ...
    def clear(self) -> None: ...
    def pop(self, key: str) -> Module: ...
    def keys(self) -> Iterable[str]: ...
    def items(self) -> Iterable[Tuple[str, Module]]: ...
    def values(self) -> Iterable[Module]: ...
    def update(self, modules: Mapping[str, Module]) -> None: ...

class ParameterList(Module):
    def __init__(self, values: Optional[Iterable[Any]] = ...) -> None: ...
    @overload
    def __getitem__(self, idx: int) -> Any: ...
    @overload
    def __getitem__(self, idx: slice) -> T: ...
    def __setitem__(self, idx: int, param: Any) -> None: ...
    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[Any]: ...
    def __iadd__(self, parameters: Iterable[Any]) -> ParameterList: ...
    def __dir__(self): ...
    def append(self, value: Any) -> ParameterList: ...
    def extend(self, values: Iterable[Any]) -> ParameterList: ...
    def extra_repr(self) -> str: ...
    def __call__(self, *args, **kwargs) -> None: ...

class ParameterDict(Module):
    def __init__(self, parameters: Any = ...) -> None: ...
    def __getitem__(self, key: str) -> Any: ...
    def __setitem__(self, key: str, value: Any) -> None: ...
    def __delitem__(self, key: str) -> None: ...
    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[str]: ...
    def __reversed__(self) -> Iterator[str]: ...
    def copy(self) -> ParameterDict: ...
    def __contains__(self, key: str) -> bool: ...
    def setdefault(self, key: str, default: Optional[Any] = ...) -> Any: ...
    def clear(self) -> None: ...
    def pop(self, key: str) -> Any: ...
    def popitem(self) -> Tuple[str, Any]: ...
    def get(self, key: str, default: Optional[Any] = ...) -> Any: ...
    def fromkeys(self, keys: Iterable[str], default: Optional[Any] = ...) -> ParameterDict: ...
    def keys(self) -> Iterable[str]: ...
    def items(self) -> Iterable[Tuple[str, Any]]: ...
    def values(self) -> Iterable[Any]: ...
    def update(self, parameters: Union[Mapping[str, Any], 'ParameterDict']) -> None: ...
    def extra_repr(self) -> str: ...
    def __call__(self, input) -> None: ...
    def __or__(self, other: ParameterDict) -> ParameterDict: ...
    def __ror__(self, other: ParameterDict) -> ParameterDict: ...
    def __ior__(self, other: ParameterDict) -> ParameterDict: ...