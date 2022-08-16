from typing import Dict, Iterator, Union

import praw

from ...base import PRAWBase as PRAWBase
from ..generator import ListingGenerator as ListingGenerator

class RisingListingMixin(PRAWBase):
    def random_rising(self, **generator_kwargs: Union[str, int, Dict[str, str]]) -> Iterator['praw.models.Submission']: ...
    def rising(self, **generator_kwargs: Union[str, int, Dict[str, str]]) -> Iterator['praw.models.Submission']: ...
