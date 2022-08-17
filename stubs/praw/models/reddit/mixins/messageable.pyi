# pylint: disable=import-error,relative-beyond-top-level,unused-import
# pylint: disable=useless-import-alias,multiple-statements,no-name-in-module
# pylint: disable=unused-argument,invalid-name
# pylint: disable=too-few-public-methods
from typing import Optional, Union

import praw

from ....const import API_PATH as API_PATH

class MessageableMixin:
    def message(
        self, *, from_subreddit: Optional[
            Union['praw.models.Subreddit', str]] = ...,
        message: str, subject: str) -> None: ...
