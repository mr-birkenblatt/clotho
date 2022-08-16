from typing import Any, Optional

from ..base import PRAWBase as PRAWBase

class Listing(PRAWBase):
    CHILD_ATTRIBUTE: str
    def __len__(self) -> int: ...
    def __getitem__(self, index: int) -> Any: ...
    def __setattr__(self, attribute: str, value: Any) -> None: ...

class FlairListing(Listing):
    CHILD_ATTRIBUTE: str
    @property
    def after(self) -> Optional[Any]: ...

class ModeratorListing(Listing):
    CHILD_ATTRIBUTE: str

class ModNoteListing(Listing):
    CHILD_ATTRIBUTE: str
    @property
    def after(self) -> Optional[Any]: ...

class ModmailConversationsListing(Listing):
    CHILD_ATTRIBUTE: str
    @property
    def after(self) -> Optional[str]: ...
