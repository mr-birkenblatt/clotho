import re
from typing import Optional

VALID_NAME = re.compile(r"^[a-z0-9_]+$")


class User:
    def __init__(self, name: str) -> None:
        if VALID_NAME.search(name) is None:
            raise ValueError(f"invalid user name {name}")
        self._name = name

    def get_name(self) -> str:
        return self._name

    def can_create_topic(self) -> bool:
        # FIXME implement permissiosn
        return True

    @staticmethod
    def parse_name(name: str) -> 'User':
        return User(name)

    def get_reputation(self) -> float:
        # FIXME implement ELO
        return 1000.0

    def get_encounters(
            self, other: 'User') -> int:  # pylint: disable=unused-argument
        # FIXME: implement encounters
        return 0

    def get_weighted_vote(
            self,
            owner: Optional['User'],  # pylint: disable=unused-argument
            ) -> float:
        # FIXME: implement ELO
        return 1.0

    def __hash__(self) -> int:
        return hash(self.get_name())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False
        if self is other:
            return True
        return self.get_name() == other.get_name()

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    def __str__(self) -> str:
        return self.get_name()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}[{self.get_name()}]"
