from typing import Iterable

import torch

from model.embedding import EmbeddingProvider
from system.msgs.message import Message, MHash
from system.msgs.store import MessageStore


class EmbeddingStore:
    def __init__(self, providers: list[EmbeddingProvider]) -> None:
        self._providers = {
            provider.get_name(): provider
            for provider in providers
        }
        assert len(self._providers) == len(providers)

    def get_names(self) -> list[str]:
        return list(self._providers.keys())

    def get_provider(self, name: str) -> EmbeddingProvider:
        return self._providers[name]

    def do_add_embedding(
            self,
            name: str,
            mhash: MHash,
            embed: torch.Tensor) -> None:
        raise NotImplementedError()

    def add_embedding(self, name: str, msg: Message) -> torch.Tensor:
        provider = self._providers[name]
        embed = provider.get_embedding(msg)
        self.do_add_embedding(name, msg.get_hash(), embed)
        return embed

    def do_get_embedding(
            self,
            name: str,
            mhash: MHash) -> torch.Tensor | None:
        raise NotImplementedError()

    def get_embedding(
            self,
            msg_store: MessageStore,
            name: str,
            mhash: MHash) -> torch.Tensor:
        if name not in self._providers:
            raise ValueError(f"{name} not found in {self._providers}")
        res = self.do_get_embedding(name, mhash)
        if res is not None:
            return res
        msg = msg_store.read_message(mhash)
        return self.add_embedding(name, msg)

    def ensure_all(
            self,
            msg_store: MessageStore,
            names: list[str] | None = None) -> None:
        if names is None:
            names = self.get_names()
        for mhash in msg_store.enumerate_messages():
            for name in names:
                self.get_embedding(msg_store, name, mhash)

    def do_get_closest(
            self, name: str, embed: torch.Tensor) -> Iterable[MHash]:
        raise NotImplementedError()

    def get_closest(self, name: str, embed: torch.Tensor) -> Iterable[MHash]:
        yield from self.do_get_closest(name, embed)

    def get_closest_for_hash(
            self,
            msg_store: MessageStore,
            name: str,
            mhash: MHash) -> Iterable[MHash]:
        yield from self.do_get_closest(
            name, self.get_embedding(msg_store, name, mhash))
