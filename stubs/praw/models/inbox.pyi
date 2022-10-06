# pylint: disable=import-error,relative-beyond-top-level,unused-import
# pylint: disable=useless-import-alias,multiple-statements,no-name-in-module
# pylint: disable=unused-argument,invalid-name
from typing import Dict, Iterator, List, Union

import praw

from ..const import API_PATH as API_PATH
from .base import PRAWBase as PRAWBase
from .listing.generator import ListingGenerator as ListingGenerator
from .util import stream_generator as stream_generator


class Inbox(PRAWBase):
    def all(
        self, **generator_kwargs: Union[str, int, Dict[str, str]],
        ) -> Iterator[Union['praw.models.Message', 'praw.models.Comment']]: ...

    def collapse(self, items: List['praw.models.Message']) -> None: ...

    def comment_replies(
        self, **generator_kwargs: Union[str, int, Dict[str, str]],
        ) -> Iterator['praw.models.Comment']: ...

    def mark_all_read(self) -> None: ...

    def mark_read(
        self,
        items: List[Union['praw.models.Comment', 'praw.models.Message']],
        ) -> None: ...

    def mark_unread(
        self,
        items: List[Union['praw.models.Comment', 'praw.models.Message']],
        ) -> None: ...

    def mentions(
        self, **generator_kwargs: Union[str, int, Dict[str, str]],
        ) -> Iterator['praw.models.Comment']: ...

    def message(self, message_id: str) -> praw.models.Message: ...

    def messages(
        self, **generator_kwargs: Union[str, int, Dict[str, str]],
        ) -> Iterator['praw.models.Message']: ...

    def sent(
        self, **generator_kwargs: Union[str, int, Dict[str, str]],
        ) -> Iterator['praw.models.Message']: ...

    def stream(
        self, **stream_options: Union[str, int, Dict[str, str]],
        ) -> Iterator[Union['praw.models.Comment', 'praw.models.Message']]: ...

    def submission_replies(
        self, **generator_kwargs: Union[str, int, Dict[str, str]],
        ) -> Iterator['praw.models.Comment']: ...

    def uncollapse(self, items: List['praw.models.Message']) -> None: ...

    def unread(
        self, *, mark_read: bool = ...,
        **generator_kwargs: Union[str, int, Dict[str, str]],
        ) -> Iterator[Union['praw.models.Comment', 'praw.models.Message']]: ...
