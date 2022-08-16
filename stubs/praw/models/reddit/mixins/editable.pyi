from typing import Union

import praw

from ....const import API_PATH as API_PATH

class EditableMixin:
    def delete(self) -> None: ...
    def edit(self, *, body: str) -> Union['praw.models.Comment', 'praw.models.Submission']: ...
