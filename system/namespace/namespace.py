from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from system.links.store import LinkModule
    from system.msgs.store import MsgsModule
    from system.namespace.load import NamespaceObj
    from system.suggest.suggest import SuggestModule
    from system.users.store import UsersModule


class Namespace:
    def __init__(self, name: str, obj: 'NamespaceObj') -> None:
        self._name = name
        self._obj = obj

    def get_name(self) -> str:
        return self._name

    def get_message_module(self) -> 'MsgsModule':
        return self._obj["msgs"]

    def get_link_module(self) -> 'LinkModule':
        return self._obj["links"]

    def get_suggest_module(self) -> 'SuggestModule':
        return self._obj["suggest"]

    def get_users_module(self) -> 'UsersModule':
        return self._obj["users"]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False
        if self is other:
            return True
        return self.get_name() == other.get_name()

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    def __hash__(self) -> int:
        return hash(self.get_name())
