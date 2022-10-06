# pylint: disable=import-error,relative-beyond-top-level,unused-import
# pylint: disable=useless-import-alias,multiple-statements,no-name-in-module
# pylint: disable=unused-argument,invalid-name,too-few-public-methods
from typing import Any, Dict, Iterator, Union

import praw

from ....util.cache import cachedproperty as cachedproperty
from ..generator import ListingGenerator as ListingGenerator
from .base import BaseListingMixin as BaseListingMixin
from .gilded import GildedListingMixin as GildedListingMixin


class SubListing(BaseListingMixin):
    def __init__(
        self, reddit: praw.Reddit, base_path: str, subpath: str) -> None: ...


class RedditorListingMixin(BaseListingMixin, GildedListingMixin):
    def comments(self) -> SubListing: ...

    def submissions(self) -> SubListing: ...

    def downvoted(
        self, **generator_kwargs: Union[str, int, Dict[str, str]],
        ) -> Iterator[Any]: ...

    def gildings(
        self, **generator_kwargs: Union[str, int, Dict[str, str]],
        ) -> Iterator[Any]: ...

    def hidden(
        self, **generator_kwargs: Union[str, int, Dict[str, str]],
        ) -> Iterator[Any]: ...

    def saved(
        self, **generator_kwargs: Union[str, int, Dict[str, str]],
        ) -> Iterator[Any]: ...

    def upvoted(
        self, **generator_kwargs: Union[str, int, Dict[str, str]],
        ) -> Iterator[Any]: ...
