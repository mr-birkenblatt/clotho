# pylint: disable=import-error,relative-beyond-top-level,unused-import
# pylint: disable=useless-import-alias,multiple-statements,no-name-in-module
# pylint: disable=unused-argument,invalid-name
from typing import Any, Dict, Iterator, List, Optional, Union

from ..const import API_PATH as API_PATH
from . import Subreddit as Subreddit
from .base import PRAWBase as PRAWBase
from .listing.generator import ListingGenerator as ListingGenerator
from .reddit.subreddit import Subreddit
from .util import stream_generator as stream_generator


class Subreddits(PRAWBase):
    def default(
        self, **generator_kwargs: Union[str, int, Dict[str, str]],
        ) -> Iterator[Subreddit]: ...

    def gold(
        self, **generator_kwargs: Any,
        ) -> Iterator[Subreddit]: ...

    def premium(
        self, **generator_kwargs: Union[str, int, Dict[str, str]],
        ) -> Iterator[Subreddit]: ...

    def new(
        self, **generator_kwargs: Union[str, int, Dict[str, str]],
        ) -> Iterator[Subreddit]: ...

    def popular(
        self, **generator_kwargs: Union[str, int, Dict[str, str]],
        ) -> Iterator[Subreddit]: ...

    def recommended(
        self, subreddits: List[Union[str, Subreddit]],
        omit_subreddits: Optional[
            List[Union[str, Subreddit]]] = ...,
        ) -> List[Subreddit]: ...

    def search(
        self, query: str,
        **generator_kwargs: Union[str, int, Dict[str, str]],
        ) -> Iterator[Subreddit]: ...

    def search_by_name(
        self, query: str, *, include_nsfw: bool = ...,
        exact: bool = ...) -> List[Subreddit]: ...

    def search_by_topic(self, query: str) -> List[Subreddit]: ...

    def stream(
        self, **stream_options: Union[str, int, Dict[str, str]],
        ) -> Iterator[Subreddit]: ...
