import os
from typing import TYPE_CHECKING

from misc.env import envload_path


if TYPE_CHECKING:
    from model.embedding import EmbeddingProviderModule
    from system.embedding.store import EmbedModule
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

    @staticmethod
    def get_root_for(ns_name: str) -> str:
        base_path = envload_path("USER_PATH", default="userdata")
        return os.path.abspath(os.path.join(base_path, ns_name))

    def get_root(self) -> str:
        return self.get_root_for(self._name)

    def get_message_module(self) -> 'MsgsModule':
        return self._obj["msgs"]

    def get_link_module(self) -> 'LinkModule':
        return self._obj["links"]

    def get_suggest_module(self) -> list['SuggestModule']:
        return self._obj["suggest"]

    def get_users_module(self) -> 'UsersModule':
        return self._obj["users"]

    def get_embed_module(self) -> 'EmbedModule':
        return self._obj["embed"]

    def get_embedding_providers(self) -> 'EmbeddingProviderModule':
        return self._obj["model"]

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
