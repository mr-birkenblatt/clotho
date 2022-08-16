# pylint: disable=import-error,relative-beyond-top-level,unused-import
# pylint: disable=useless-import-alias,multiple-statements
from typing import Any, Dict, List, Optional, Union

import praw
from _typeshed import Incomplete

from .exceptions import ClientException as ClientException
from .exceptions import RedditAPIException as RedditAPIException
from .models.reddit.base import RedditBase as RedditBase
from .util import snake_case_keys as snake_case_keys

class Objector:
    @classmethod
    def parse_error(cls, data: Union[List[Any], Dict[str, Dict[str, str]]]) -> Optional[RedditAPIException]: ...
    @classmethod
    def check_error(cls, data: Union[List[Any], Dict[str, Dict[str, str]]]) -> None: ...
    parsers: Incomplete
    def __init__(self, reddit: praw.Reddit, parsers: Optional[Dict[str, Any]] = ...) -> None: ...
    def objectify(self, data: Optional[Union[Dict[str, Any], List[Any], bool]]) -> Optional[Union[RedditBase, Dict[str, Any], List[Any], bool]]: ...
