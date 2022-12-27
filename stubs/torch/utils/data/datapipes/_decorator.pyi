# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


from typing import Any, Callable, Optional, Type, Union

from _typeshed import Incomplete
from torch.utils.data.datapipes.datapipe import IterDataPipe as IterDataPipe
from torch.utils.data.datapipes.datapipe import MapDataPipe as MapDataPipe


class functional_datapipe:
    name: str
    enable_df_api_tracing: Incomplete

    def __init__(
        self, name: str, enable_df_api_tracing: bool = ...) -> None: ...

    def __call__(self, cls): ...


class guaranteed_datapipes_determinism:
    prev: bool
    def __init__(self) -> None: ...
    def __enter__(self) -> None: ...

    def __exit__(
        self, exc_type: Any, exc_value: Any, traceback: Any) -> None: ...


class non_deterministic:
    cls: Optional[Type[IterDataPipe]]
    deterministic_fn: Callable[[], bool]

    def __init__(
        self, arg: Union[Type[IterDataPipe], Callable[[], bool]]) -> None: ...

    def __call__(self, *args, **kwargs): ...
    def deterministic_wrapper_fn(self, *args, **kwargs) -> IterDataPipe: ...


def argument_validation(f): ...


class runtime_validation_disabled:
    prev: bool
    def __init__(self) -> None: ...
    def __enter__(self) -> None: ...

    def __exit__(
        self, exc_type: Any, exc_value: Any, traceback: Any) -> None: ...


def runtime_validation(f): ...
