# pylint: disable=import-error,relative-beyond-top-level,unused-import
# pylint: disable=useless-import-alias,multiple-statements,no-name-in-module
# pylint: disable=unused-argument
from typing import List, Optional, Union

import praw

from ..exceptions import DuplicateReplaceException as DuplicateReplaceException
from .reddit.more import MoreComments as MoreComments


class CommentForest:
    def __getitem__(
        self,
        index: int) -> Union[
            'praw.models.Comment', 'praw.models.MoreComments']: ...

    def __init__(
        self, submission: praw.models.Submission,
        comments: Optional[List['praw.models.Comment']] = ...) -> None: ...

    def __len__(self) -> int: ...

    def list(self) -> List[
        Union['praw.models.Comment', 'praw.models.MoreComments']]: ...

    def replace_more(
        self, *, limit: int = ...,
        threshold: int = ...) -> List['praw.models.MoreComments']: ...
