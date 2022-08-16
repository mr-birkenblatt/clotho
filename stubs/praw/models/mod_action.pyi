from typing import Union

import praw

from .base import PRAWBase as PRAWBase

class ModAction(PRAWBase):
    @property
    def mod(self) -> praw.models.Redditor: ...
    @mod.setter
    def mod(self, value: Union[str, 'praw.models.Redditor']): ...
