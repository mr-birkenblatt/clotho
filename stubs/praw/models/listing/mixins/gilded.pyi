# pylint: disable=import-error,relative-beyond-top-level,unused-import
# pylint: disable=useless-import-alias,multiple-statements,no-name-in-module
# pylint: disable=unused-argument,invalid-name,too-few-public-methods
from typing import Any, Dict, Iterator, Union

from ...base import PRAWBase as PRAWBase
from ..generator import ListingGenerator as ListingGenerator

class GildedListingMixin(PRAWBase):
    def gilded(
        self, **generator_kwargs: Union[str, int, Dict[str, str]],
        ) -> Iterator[Any]: ...
