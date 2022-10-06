# pylint: disable=import-error,relative-beyond-top-level,unused-import
# pylint: disable=useless-import-alias,multiple-statements,no-name-in-module
# pylint: disable=unused-argument,invalid-name
from typing import Any, Dict, Optional, Union

import praw

from ...exceptions import InvalidURL as InvalidURL
from ..base import PRAWBase as PRAWBase


class RedditBase(PRAWBase):
    def __eq__(self, other: Union[Any, str]) -> bool: ...
    def __getattr__(self, attribute: str) -> Any: ...
    def __hash__(self) -> int: ...

    def __init__(
        self, reddit: praw.Reddit, _data: Optional[Dict[str, Any]],
        _extra_attribute_to_check: Optional[str] = ..., _fetched: bool = ...,
        _str_field: bool = ...) -> None: ...

    def __ne__(self, other: Any) -> bool: ...
