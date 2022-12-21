from typing import Iterable

import torch

from model.embedding import EmbeddingProvider
from system.msgs.message import Message
from system.msgs.store import MessageStore


class EmbeddingStore:
    def __init__(self, providers: list[EmbeddingProvider]) -> None:
        self._providers = {
            provider.get_name(): provider
            for provider in providers
        }
        assert len(self._providers) == len(providers)

    def get_provider(self, name: str) -> EmbeddingProvider:
        return self._providers[name]

    def do_add_embedding(
            self,
            name: str,
            msg: Message,
            embed: torch.Tensor) -> None:
        raise NotImplementedError()

    def add_embedding(self, name: str, msg: Message) -> torch.Tensor:
        provider = self._providers[name]
        embed = provider.get_embedding(msg)
        self.do_add_embedding(name, msg, embed)
        return embed

    def do_get_embedding(
            self,
            name: str,
            msg: Message) -> torch.Tensor | None:
        raise NotImplementedError()

    def get_embedding(self, name: str, msg: Message) -> torch.Tensor:
        if name not in self._providers:
            raise ValueError(f"{name} not found in {self._providers}")
        res = self.do_get_embedding(name, msg)
        if res is not None:
            return res
        return self.add_embedding(name, msg)

    def ensure_all(self, msg_store: MessageStore, names: list[str]) -> None:
        for msg in msg_store.enumerate_messages():
            for name in names:
                self.get_embedding(name, msg)

    def do_get_closest(self, name: str, msg: Message) -> Iterable[Message]:
        raise NotImplementedError()

    def get_closest(self, name: str, msg: Message) -> Iterable[Message]:
        yield from self.do_get_closest(name, msg)
