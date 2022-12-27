# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access,no-name-in-module,undefined-variable


from typing import Callable, Dict, Iterator, Optional, TypeVar

from _typeshed import Incomplete
from torch.utils.data.datapipes._typing import _DataPipeMeta, _IterDataPipeMeta
from torch.utils.data.dataset import Dataset, IterableDataset


T = TypeVar('T')
T_co = TypeVar('T_co', covariant=True)


class IterDataPipe(IterableDataset[T_co], metaclass=_IterDataPipeMeta):
    functions: Dict[str, Callable]
    reduce_ex_hook: Optional[Callable]
    getstate_hook: Optional[Callable]
    str_hook: Optional[Callable]
    repr_hook: Optional[Callable]
    def __getattr__(self, attribute_name): ...
    @classmethod
    def register_function(cls, function_name, function) -> None: ...

    @classmethod
    def register_datapipe_as_function(
        cls, function_name, cls_to_register,
        enable_df_api_tracing: bool = ...): ...

    def __reduce_ex__(self, *args, **kwargs): ...
    @classmethod
    def set_getstate_hook(cls, hook_fn) -> None: ...
    @classmethod
    def set_reduce_ex_hook(cls, hook_fn) -> None: ...
    def reset(self) -> None: ...


class DFIterDataPipe(IterDataPipe):
    ...


class MapDataPipe(Dataset[T_co], metaclass=_DataPipeMeta):
    functions: Dict[str, Callable]
    reduce_ex_hook: Optional[Callable]
    getstate_hook: Optional[Callable]
    str_hook: Optional[Callable]
    repr_hook: Optional[Callable]
    def __getattr__(self, attribute_name): ...
    @classmethod
    def register_function(cls, function_name, function) -> None: ...
    @classmethod
    def register_datapipe_as_function(cls, function_name, cls_to_register): ...
    def __reduce_ex__(self, *args, **kwargs): ...
    @classmethod
    def set_getstate_hook(cls, hook_fn) -> None: ...
    @classmethod
    def set_reduce_ex_hook(cls, hook_fn) -> None: ...


class _DataPipeSerializationWrapper:
    def __init__(self, datapipe) -> None: ...
    def __len__(self): ...


class _IterDataPipeSerializationWrapper(
        _DataPipeSerializationWrapper, IterDataPipe):
    def __iter__(self): ...


class _MapDataPipeSerializationWrapper(
        _DataPipeSerializationWrapper, MapDataPipe):
    def __getitem__(self, idx): ...


class DataChunk(list):
    items: Incomplete
    def __init__(self, items) -> None: ...
    def as_str(self, indent: str = ...): ...
    def __iter__(self) -> Iterator[T]: ...
    def raw_iterator(self) -> T: ...
