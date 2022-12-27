from contextlib import contextmanager
from typing import Iterable, Iterator, Literal, TypedDict

import torch

from misc.redis import create_redis_config, get_redis_ns_key, register_redis_ns
from model.embedding import EmbeddingProvider, get_embed_providers
from system.msgs.message import Message, MHash
from system.msgs.store import MessageStore
from system.namespace.namespace import Namespace


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

    @contextmanager
    def bulk_add(self, name: str) -> Iterator[None]:
        raise NotImplementedError()

    def ensure_all(
            self,
            msg_store: MessageStore,
            names: list[str] | None = None) -> None:
        if names is None:
            names = self.get_names()
        for name in names:
            with self.bulk_add(name):
                # FIXME: remove stop early
                for mhash in list(msg_store.enumerate_messages())[:20]:
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


EMBED_STORE: dict[Namespace, EmbeddingStore] = {}


def get_embed_store(namespace: Namespace) -> EmbeddingStore:
    res = EMBED_STORE.get(namespace)
    if res is None:
        res = create_embed_store(namespace)
        EMBED_STORE[namespace] = res
    return res


RedisEmbedModule = TypedDict('RedisEmbedModule', {
    "name": Literal["redis"],
    "host": str,
    "port": int,
    "passwd": str,
    "prefix": str,
    "path": str,
    "index": Literal["annoy"],
})
NoEmbedModule = TypedDict('NoEmbedModule', {
    "name": Literal["none"],
})
EmbedModule = RedisEmbedModule | NoEmbedModule


def create_embed_store(namespace: Namespace) -> EmbeddingStore:
    eobj = namespace.get_embed_module()
    providers = get_embed_providers(namespace)
    if eobj["name"] == "redis":
        from system.embedding.annoy import AnnoyEmbeddingStore
        from system.embedding.rediscache import RedisEmbeddingCache

        ns_key = get_redis_ns_key(namespace.get_name(), "embedding")
        cache = RedisEmbeddingCache(ns_key)
        if eobj["index"] != "annoy":
            raise ValueError(f"unsupported embedding index: {eobj['index']}")
        if not ns_key[0].startswith("_"):
            register_redis_ns(ns_key, create_redis_config(
                eobj["host"],
                eobj["port"],
                eobj["passwd"],
                eobj["prefix"],
                eobj["path"]))
        return AnnoyEmbeddingStore(providers, cache, eobj["path"])
    if eobj["name"] == "none":
        from system.embedding.noembed import NoEmbedding

        return NoEmbedding(providers)
    raise ValueError(f"unknown embed store: {eobj}")
