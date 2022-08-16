from typing import Iterator, Union

import praw

from .listing.generator import ListingGenerator as ListingGenerator
from .listing.mixins import SubredditListingMixin as SubredditListingMixin

class Front(SubredditListingMixin):
    def __init__(self, reddit: praw.Reddit) -> None: ...
    def best(self, **generator_kwargs: Union[str, int]) -> Iterator['praw.models.Submission']: ...
