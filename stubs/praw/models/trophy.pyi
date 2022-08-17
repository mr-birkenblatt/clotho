# pylint: disable=import-error,relative-beyond-top-level,unused-import
# pylint: disable=useless-import-alias,multiple-statements,no-name-in-module
# pylint: disable=unused-argument,invalid-name
from typing import Any, Dict, Union

import praw

from .base import PRAWBase as PRAWBase

class Trophy(PRAWBase):
    def __init__(self, reddit: praw.Reddit, _data: Dict[str, Any]) -> None: ...
    def __eq__(self, other: Union['Trophy', Any]) -> bool: ...
