# pylint: disable=import-error,relative-beyond-top-level,unused-import
# pylint: disable=useless-import-alias,multiple-statements,no-name-in-module
# pylint: disable=unused-argument,invalid-name,too-few-public-methods
from typing import Any, Dict, Optional

import praw

class PRAWBase:
    @classmethod
    def parse(cls, data: Dict[str, Any], reddit: praw.Reddit) -> Any: ...

    def __init__(
        self, reddit: praw.Reddit,
        _data: Optional[Dict[str, Any]]) -> None: ...
