# pylint: disable=import-error,relative-beyond-top-level,unused-import
# pylint: disable=useless-import-alias,multiple-statements,no-name-in-module
# pylint: disable=unused-argument,invalid-name
from typing import Any, Dict, Iterator, Union

from _typeshed import Incomplete

from ...base import PRAWBase as PRAWBase
from ...reddit.submission import Submission
from ..generator import ListingGenerator as ListingGenerator


class BaseListingMixin(PRAWBase):
    VALID_TIME_FILTERS: Incomplete

    def controversial(
        self, *, time_filter: str = ...,
        **generator_kwargs: Union[str, int, Dict[str, str]],
        ) -> Iterator[Submission]: ...

    def hot(
        self, **generator_kwargs: Union[str, int, Dict[str, str]],
        ) -> Iterator[Submission]: ...

    def new(
        self, **generator_kwargs: Union[str, int, Dict[str, str]],
        ) -> Iterator[Submission]: ...

    def top(
        self, *, time_filter: str = ...,
        **generator_kwargs: Union[str, int, Dict[str, str]],
        ) -> Iterator[Submission]: ...
