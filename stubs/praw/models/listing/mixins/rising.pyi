# pylint: disable=import-error,relative-beyond-top-level,unused-import
# pylint: disable=useless-import-alias,multiple-statements,no-name-in-module
# pylint: disable=unused-argument,invalid-name
from typing import Dict, Iterator, Union

import praw

from ...base import PRAWBase as PRAWBase
from ..generator import ListingGenerator as ListingGenerator


class RisingListingMixin(PRAWBase):
    def random_rising(
        self,
        **generator_kwargs: Union[str, int, Dict[str, str]],
        ) -> Iterator['praw.models.Submission']: ...

    def rising(
        self, **generator_kwargs: Union[str, int, Dict[str, str]],
        ) -> Iterator['praw.models.Submission']: ...
