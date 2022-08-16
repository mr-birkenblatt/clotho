# pylint: disable=import-error,relative-beyond-top-level,unused-import
# pylint: disable=useless-import-alias,multiple-statements,no-name-in-module
# pylint: disable=unused-argument,invalid-name
from types import SimpleNamespace
from typing import Dict, Iterable, Iterator, Union

import praw

from ..const import API_PATH as API_PATH
from .base import PRAWBase as PRAWBase
from .listing.generator import ListingGenerator as ListingGenerator
from .util import stream_generator as stream_generator

class PartialRedditor(SimpleNamespace):
    ...


class Redditors(PRAWBase):
    def new(
        self, **generator_kwargs: Union[str, int, Dict[str, str]],
        ) -> Iterator['praw.models.Subreddit']: ...

    def popular(
        self, **generator_kwargs: Union[str, int, Dict[str, str]],
        ) -> Iterator['praw.models.Subreddit']: ...

    def search(
        self, query: str, **generator_kwargs: Union[str, int, Dict[str, str]],
        ) -> Iterator['praw.models.Subreddit']: ...

    def stream(
        self, **stream_options: Union[str, int, Dict[str, str]],
        ) -> Iterator['praw.models.Subreddit']: ...

    def partial_redditors(
        self, ids: Iterable[str]) -> Iterator[PartialRedditor]: ...
