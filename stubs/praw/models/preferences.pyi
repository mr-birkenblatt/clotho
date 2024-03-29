# pylint: disable=import-error,relative-beyond-top-level,unused-import
# pylint: disable=useless-import-alias,multiple-statements,no-name-in-module
# pylint: disable=unused-argument,invalid-name
from typing import Dict, Union

import praw

from ..const import API_PATH as API_PATH


class Preferences:
    def __call__(self) -> Dict[str, Union[bool, int, str]]: ...
    def __init__(self, reddit: praw.Reddit) -> None: ...
    def update(self, **preferences: Union[bool, int, str]) -> None: ...
