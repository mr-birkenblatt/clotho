# pylint: disable=import-error,relative-beyond-top-level,unused-import
# pylint: disable=useless-import-alias,multiple-statements,no-name-in-module
# pylint: disable=unused-argument
from typing import Iterator, List, Optional, Union

from ..exceptions import DuplicateReplaceException as DuplicateReplaceException
from .reddit.comment import Comment as Comment
from .reddit.more import MoreComments as MoreComments
from .reddit.submission import Submission as Submission


class CommentForest:
    def __getitem__(
        self,
        index: int) -> Union[Comment, MoreComments]: ...

    def __init__(
        self, submission: Submission,
        comments: Optional[List[Comment]] = ...) -> None: ...

    def __iter__(self) -> Iterator[Union[Comment, MoreComments]]:
        ...

    def __len__(self) -> int: ...

    def list(self) -> List[Union[Comment, MoreComments]]: ...

    def replace_more(
        self, *, limit: int = ...,
        threshold: int = ...) -> List[MoreComments]: ...
