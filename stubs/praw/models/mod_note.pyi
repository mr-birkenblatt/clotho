# pylint: disable=import-error,relative-beyond-top-level,unused-import
# pylint: disable=useless-import-alias,multiple-statements,no-name-in-module
# pylint: disable=unused-argument,invalid-name
from praw.endpoints import API_PATH as API_PATH

from .base import PRAWBase as PRAWBase

class ModNote(PRAWBase):
    def __eq__(self, other: object) -> bool: ...
    def delete(self) -> None: ...
