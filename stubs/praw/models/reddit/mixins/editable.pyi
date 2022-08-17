# pylint: disable=import-error,relative-beyond-top-level,unused-import
# pylint: disable=useless-import-alias,multiple-statements,no-name-in-module
# pylint: disable=unused-argument,invalid-name
from typing import Union

import praw

from ....const import API_PATH as API_PATH

class EditableMixin:
    def delete(self) -> None: ...

    def edit(
        self, *, body: str,
        ) -> Union['praw.models.Comment', 'praw.models.Submission']: ...
