from typing import Any, Dict, Iterator, Union

from ...base import PRAWBase as PRAWBase
from ..generator import ListingGenerator as ListingGenerator

class GildedListingMixin(PRAWBase):
    def gilded(self, **generator_kwargs: Union[str, int, Dict[str, str]]) -> Iterator[Any]: ...
