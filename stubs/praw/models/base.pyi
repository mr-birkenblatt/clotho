from typing import Any, Dict, Optional

import praw

class PRAWBase:
    @classmethod
    def parse(cls, data: Dict[str, Any], reddit: praw.Reddit) -> Any: ...

    def __init__(
        self, reddit: praw.Reddit,
        _data: Optional[Dict[str, Any]]) -> None: ...
