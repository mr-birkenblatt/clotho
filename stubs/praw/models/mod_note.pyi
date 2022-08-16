from praw.endpoints import API_PATH as API_PATH

from .base import PRAWBase as PRAWBase

class ModNote(PRAWBase):
    def __eq__(self, other): ...
    def delete(self) -> None: ...
