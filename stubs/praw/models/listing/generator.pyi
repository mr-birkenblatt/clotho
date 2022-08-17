# pylint: disable=import-error,relative-beyond-top-level,unused-import
# pylint: disable=useless-import-alias,multiple-statements,no-name-in-module
# pylint: disable=unused-argument,invalid-name
from typing import Any, Dict, Iterator, Optional, Union

import praw
from _typeshed import Incomplete

from ..base import PRAWBase as PRAWBase
from .listing import FlairListing as FlairListing
from .listing import ModNoteListing as ModNoteListing

class ListingGenerator(PRAWBase, Iterator):
    limit: Incomplete
    params: Incomplete
    url: Incomplete
    yielded: int

    def __init__(
        self, reddit: praw.Reddit, url: str, limit: int = ...,
        params: Optional[Dict[str, Union[str, int]]] = ...) -> None: ...

    def __iter__(self) -> Iterator[Any]: ...
    def __next__(self) -> Any: ...
