# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access,no-name-in-module,undefined-variable


from typing import Callable, List, Optional, TypeVar, Union

import torch


T = TypeVar('T')
S = TypeVar('S')


class _PyFutureMeta:
    ...


class Future(torch._C.Future, metaclass=_PyFutureMeta):

    def __init__(
        self, *, devices: Optional[List[Union[int, str,
                                torch.device]]] = ...) -> None: ...

    def done(self) -> bool: ...
    def wait(self) -> T: ...
    def value(self) -> T: ...
    def then(self, callback: Callable[[Future[T]], S]) -> Future[S]: ...

    def add_done_callback(
        self, callback: Callable[[Future[T]], None]) -> None: ...

    def set_result(self, result: T) -> None: ...
    def set_exception(self, result: T) -> None: ...


def collect_all(futures: List[Future]) -> Future[List[Future]]: ...


def wait_all(futures: List[Future]) -> List: ...
