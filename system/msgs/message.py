import re
from typing import Optional

from misc.util import get_text_hash, is_hex

TOPIC_START = "t/"
VALID_TOPIC = re.compile(r"^t\/[a-z0-9_]+$")


class MHash:
    def __init__(self, msg_hash: str) -> None:
        self._hash = msg_hash

    @classmethod
    def parse(cls, msg_hash: str) -> 'MHash':
        if len(msg_hash) != 64 or not is_hex(msg_hash):
            raise ValueError(f"cannot parse: {msg_hash}")
        return MHash(msg_hash)

    @staticmethod
    def from_message(text: str) -> 'MHash':
        return MHash(get_text_hash(text))

    def to_parseable(self) -> str:
        return self._hash

    def __hash__(self) -> int:
        return hash(self.to_parseable())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False
        if self is other:
            return True
        return self.to_parseable() == other.to_parseable()

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    def __str__(self) -> str:
        return self.to_parseable()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}[{self.to_parseable()}]"


class Message:
    def __init__(self, *, msg: str, msg_hash: Optional[MHash] = None) -> None:
        assert msg, "messages cannot be empty"
        self._msg = msg
        self._msg_hash = None
        if msg_hash is not None:
            assert MHash.from_message(msg) == msg_hash
            self._msg_hash = msg_hash

    def get_text(self) -> str:
        return self._msg

    def is_topic(self) -> bool:
        return VALID_TOPIC.search(self.get_text()) is not None

    def is_valid_message(self) -> bool:
        return not self.get_text().startswith(TOPIC_START)

    def get_hash(self) -> MHash:
        res = self._msg_hash
        if res is None:
            res = MHash.from_message(self._msg)
            self._msg_hash = res
        return res

    def __hash__(self) -> int:
        return hash(self.get_hash()) + 31 * len(self.get_text())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False
        if self is other:
            return True
        if len(self.get_text()) != len(other.get_text()):
            return False
        return self.get_hash() == other.get_hash()

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    def __str__(self) -> str:
        return f"{self.__class__.__name__}[{len(self.get_text())}]"

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}"
            f"[{len(self.get_text())},{repr(self.get_hash())}]")
