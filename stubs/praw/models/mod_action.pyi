# pylint: disable=import-error,relative-beyond-top-level,unused-import
# pylint: disable=useless-import-alias,multiple-statements,no-name-in-module
# pylint: disable=unused-argument,invalid-name
# pylint: disable=too-few-public-methods
from typing import Union

import praw

from .base import PRAWBase as PRAWBase

class ModAction(PRAWBase):
    @property
    def mod(self) -> praw.models.Redditor: ...

    @mod.setter
    def mod(self, value: Union[str, 'praw.models.Redditor']) -> None: ...
