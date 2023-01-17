import os
from contextlib import contextmanager
from typing import cast, Iterable, Iterator, Literal, TypedDict

import torch

from model.embedding import (
    EmbeddingProvider,
    EmbeddingProviderMap,
    get_embed_providers,
    ProviderRole,
)
from system.msgs.message import Message, MHash
from system.msgs.store import MessageStore
from system.namespace.module import ModuleBase
from system.namespace.namespace import ModuleName, Namespace


class EmbeddingStore(ModuleBase):
    def __init__(self, providers: EmbeddingProviderMap) -> None:
        super().__init__()
        self._providers = providers

    @staticmethod
    def module_name() -> ModuleName:
        return "embed"

    def get_roles(self) -> list[ProviderRole]:
        return cast(list[ProviderRole], list(self._providers.keys()))

    def get_provider(self, role: ProviderRole) -> EmbeddingProvider:
        return self._providers[role]

    def do_add_embedding(
            self,
            role: ProviderRole,
            mhash: MHash,
            embed: torch.Tensor) -> None:
        raise NotImplementedError()

    def add_embedding(
            self, role: ProviderRole, msg: Message) -> torch.Tensor:
        provider = self._providers[role]
        embed = provider.get_embedding(msg)
        if len(embed.shape) != 1:
            raise ValueError(f"bad embedding shape: {embed.shape}")
        self.do_add_embedding(role, msg.get_hash(), embed)
        return embed

    def do_get_embedding(
            self,
            role: ProviderRole,
            mhash: MHash) -> torch.Tensor | None:
        raise NotImplementedError()

    def get_embedding(
            self,
            msg_store: MessageStore,
            role: ProviderRole,
            mhash: MHash) -> torch.Tensor:
        if role not in self._providers:
            raise ValueError(f"{role} not found in {self._providers}")
        res = self.do_get_embedding(role, mhash)
        if res is not None:
            return res
        msg = msg_store.read_message(mhash)
        return self.add_embedding(role, msg)

    @contextmanager
    def bulk_add(self, role: ProviderRole) -> Iterator[None]:
        raise NotImplementedError()

    def ensure_all(
            self,
            msg_store: MessageStore,
            roles: list[ProviderRole] | None = None) -> None:
        if roles is None:
            roles = self.get_roles()
        for role in roles:
            with self.bulk_add(role):
                for mhash in msg_store.enumerate_messages(progress_bar=True):
                    self.get_embedding(msg_store, role, mhash)

    def self_test(self, role: ProviderRole, count: int | None) -> None:
        raise NotImplementedError()

    def do_get_closest(
            self,
            role: ProviderRole,
            embed: torch.Tensor,
            count: int,
            *,
            precise: bool) -> Iterable[MHash]:
        raise NotImplementedError()

    def get_closest(
            self,
            role: ProviderRole,
            embed: torch.Tensor,
            count: int,
            *,
            precise: bool) -> Iterable[MHash]:
        yield from self.do_get_closest(role, embed, count, precise=precise)

    def get_closest_for_hash(
            self,
            msg_store: MessageStore,
            role: ProviderRole,
            mhash: MHash,
            count: int,
            *,
            precise: bool) -> Iterable[MHash]:
        yield from self.do_get_closest(
            role,
            self.get_embedding(msg_store, role, mhash),
            count,
            precise=precise)


EMBED_STORE: dict[Namespace, EmbeddingStore] = {}


def get_embed_store(namespace: Namespace) -> EmbeddingStore:
    res = EMBED_STORE.get(namespace)
    if res is None:
        res = create_embed_store(namespace)
        EMBED_STORE[namespace] = res
    return res


RedisEmbedModule = TypedDict('RedisEmbedModule', {
    "name": Literal["redis"],
    "conn": str,
    "path": str,
    "index": Literal["annoy"],
    "trees": int,
    "shard_size": int,
    "metric": Literal["dot", "angular"],
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

        root = os.path.join(namespace.get_root(), eobj["path"])
        ns_key = namespace.get_redis_key("embedding", eobj["conn"])
        cache = RedisEmbeddingCache(ns_key)
        if eobj["index"] != "annoy":
            raise ValueError(f"unsupported embedding index: {eobj['index']}")
        return AnnoyEmbeddingStore(
            providers,
            cache,
            root,
            trees=eobj["trees"],
            shard_size=eobj["shard_size"],
            is_dot=eobj["metric"] == "dot")
    if eobj["name"] == "none":
        from system.embedding.noembed import NoEmbedding

        return NoEmbedding(providers)
    raise ValueError(f"unknown embed store: {eobj}")
