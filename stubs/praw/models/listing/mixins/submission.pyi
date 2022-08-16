from typing import Dict, Iterator, Union

import praw

from ....const import API_PATH as API_PATH
from ...base import PRAWBase as PRAWBase
from ..generator import ListingGenerator as ListingGenerator

class SubmissionListingMixin(PRAWBase):
    def duplicates(self, **generator_kwargs: Union[str, int, Dict[str, str]]) -> Iterator['praw.models.Submission']: ...
