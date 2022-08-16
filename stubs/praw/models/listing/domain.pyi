import praw

from ...const import API_PATH as API_PATH
from .mixins import BaseListingMixin as BaseListingMixin
from .mixins import RisingListingMixin as RisingListingMixin

class DomainListing(BaseListingMixin, RisingListingMixin):
    def __init__(self, reddit: praw.Reddit, domain: str) -> None: ...
