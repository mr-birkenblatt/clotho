# pylint: disable=import-error,relative-beyond-top-level,unused-import
# pylint: disable=useless-import-alias,multiple-statements
# pylint: disable=super-init-not-called
from typing import List, Optional, Union

from _typeshed import Incomplete

class PRAWException(Exception):
    ...


class RedditErrorItem:
    @property
    def error_message(self) -> str: ...
    error_type: Incomplete
    message: Incomplete
    field: Incomplete

    def __init__(
        self, error_type: str, *, field: Optional[str] = ...,
        message: Optional[str] = ...) -> None: ...

    def __eq__(
        self, other: Union['RedditErrorItem', List[str], object]) -> bool: ...


class APIException(PRAWException):
    @staticmethod
    def parse_exception_list(
        exceptions: List[Union[RedditErrorItem, List[str]]]) -> None: ...

    @property
    def error_type(self) -> str: ...
    @property
    def message(self) -> str: ...
    @property
    def field(self) -> str: ...
    items: Incomplete

    def __init__(
        self, items: Union[List[Union[RedditErrorItem, List[str], str]], str],
        *optional_args: str) -> None: ...


class RedditAPIException(APIException):
    ...


class ClientException(PRAWException):
    ...


class DuplicateReplaceException(ClientException):
    def __init__(self) -> None: ...


class InvalidFlairTemplateID(ClientException):
    def __init__(self, template_id: str) -> None: ...


class InvalidImplicitAuth(ClientException):
    def __init__(self) -> None: ...


class InvalidURL(ClientException):
    def __init__(self, url: str, *, message: str = ...) -> None: ...


class MissingRequiredAttributeException(ClientException):
    ...


class ReadOnlyException(ClientException):
    ...


class TooLargeMediaException(ClientException):
    maximum_size: Incomplete
    actual: Incomplete
    def __init__(self, *, actual: int, maximum_size: int) -> None: ...


class WebSocketException(ClientException):
    @property
    def original_exception(self) -> Exception: ...
    @original_exception.setter
    def original_exception(self, value: Exception) -> None: ...
    def __init__(
        self, message: str, exception: Optional[Exception]) -> None: ...


class MediaPostFailed(WebSocketException):
    def __init__(self) -> None: ...
