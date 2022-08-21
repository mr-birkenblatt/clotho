# pylint: disable=import-error,relative-beyond-top-level,unused-import
# pylint: disable=useless-import-alias,multiple-statements,no-name-in-module
# pylint: disable=unused-argument,invalid-name
from typing import Any, Dict, List, Union

import praw

from ...const import API_PATH as API_PATH
from ..base import PRAWBase as PRAWBase


class MoreComments(PRAWBase):
    count: int
    children: List[str]
    submission: praw.models.Submission
    def __init__(self, reddit: praw.Reddit, _data: Dict[str, Any]) -> None: ...
    def __eq__(self, other: Union[str, 'MoreComments', object]) -> bool: ...
    def __lt__(self, other: 'MoreComments') -> bool: ...

    def comments(
        self, *, update: bool = ...) -> List['praw.models.Comment']: ...
