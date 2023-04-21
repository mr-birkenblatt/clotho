import os
from typing import cast, Iterable, Literal, TYPE_CHECKING, TypedDict

import torch

from model.embedding import (
    EmbeddingProvider,
    EmbeddingProviderMap,
    get_embed_providers,
    PROVIDER_ROLES,
    ProviderRole,
)
from system.msgs.message import Message, MHash
from system.msgs.store import MessageStore
from system.namespace.module import ModuleBase
from system.namespace.namespace import ModuleName, Namespace


if TYPE_CHECKING:
    from system.embedding.index_lookup import EmbeddingCache


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
            embed: torch.Tensor,
            *,
            no_index: bool) -> None:
        raise NotImplementedError()

    def add_embedding(
            self,
            role: ProviderRole,
            msg: Message,
            *,
            no_index: bool,
            no_cache: bool) -> torch.Tensor:
        provider = self._providers[role]
        embed = provider.get_embedding(msg)
        if len(embed.shape) != 1:
            raise ValueError(f"bad embedding shape: {embed.shape}")
        if not no_cache:
            self.do_add_embedding(
                role, msg.get_hash(), embed, no_index=no_index)
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
            mhash: MHash,
            *,
            no_index: bool,
            no_cache: bool) -> torch.Tensor:
        if role not in self._providers:
            raise ValueError(f"{role} not found in {self._providers}")
        if not no_cache:
            res = self.do_get_embedding(role, mhash)
            if res is not None:
                return res
        msg = msg_store.read_message(mhash)
        return self.add_embedding(
            role, msg, no_index=no_index, no_cache=no_cache)

    def get_all_embeddings(
            self,
            role: ProviderRole,
            *,
            progress_bar: bool) -> Iterable[tuple[MHash, torch.Tensor]]:
        raise NotImplementedError()

    def get_embedding_count(self, role: ProviderRole) -> int:
        raise NotImplementedError()

    def from_namespace(
            self,
            own_namespace: Namespace,
            other_namespace: Namespace,
            *,
            progress_bar: bool) -> None:
        oembed = get_embed_store(other_namespace)
        for role in PROVIDER_ROLES:
            for mhash, embed in oembed.get_all_embeddings(
                    role, progress_bar=progress_bar):
                self.do_add_embedding(role, mhash, embed, no_index=True)

    def ensure_all(
            self,
            msg_store: MessageStore,
            roles: list[ProviderRole] | None = None,
            *,
            no_index: bool) -> None:
        if roles is None:
            roles = self.get_roles()
        for role in roles:
            for mhash in msg_store.enumerate_messages(progress_bar=True):
                self.get_embedding(
                    msg_store, role, mhash, no_index=no_index, no_cache=False)

    def self_test(self, role: ProviderRole, count: int | None) -> None:
        raise NotImplementedError()

    def do_get_closest(
            self,
            role: ProviderRole,
            embed: torch.Tensor,
            count: int,
            *,
            precise: bool,
            no_cache: bool) -> Iterable[MHash]:
        raise NotImplementedError()

    def get_closest(
            self,
            role: ProviderRole,
            embed: torch.Tensor,
            count: int,
            *,
            precise: bool,
            no_cache: bool) -> Iterable[MHash]:
        yield from self.do_get_closest(
            role, embed, count, precise=precise, no_cache=no_cache)

    def get_closest_for_hash(
            self,
            msg_store: MessageStore,
            role: ProviderRole,
            mhash: MHash,
            count: int,
            *,
            precise: bool,
            no_cache: bool) -> Iterable[MHash]:
        yield from self.do_get_closest(
            role,
            self.get_embedding(
                msg_store, role, mhash, no_index=precise, no_cache=no_cache),
            count,
            precise=precise,
            no_cache=no_cache)


EMBED_STORE: dict[Namespace, EmbeddingStore] = {}


def get_embed_store(namespace: Namespace) -> EmbeddingStore:
    res = EMBED_STORE.get(namespace)
    if res is None:
        res = create_embed_store(namespace)
        EMBED_STORE[namespace] = res
    return res


CachedEmbedModule = TypedDict('CachedEmbedModule', {
    "name": Literal["redis", "db", "cold"],
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
EmbedModule = CachedEmbedModule | NoEmbedModule


def create_embed_store(namespace: Namespace) -> EmbeddingStore:
    eobj = namespace.get_embed_module()
    providers = get_embed_providers(namespace)
    if eobj["name"] == "none":
        from system.embedding.noembed import NoEmbedding

        return NoEmbedding(providers)
    if eobj["name"] in ("redis", "db"):
        root = os.path.join(namespace.get_root(), eobj["path"])
        if eobj["name"] == "redis":
            from system.embedding.rediscache import RedisEmbeddingCache

            ns_key = namespace.get_redis_key("embedding", eobj["conn"])
            cache: 'EmbeddingCache' = RedisEmbeddingCache(ns_key)
        elif eobj["name"] == "db":
            from system.embedding.dbcache import DBEmbeddingCache

            cache = DBEmbeddingCache(
                namespace,
                namespace.get_db_connector(eobj["conn"]))
        elif eobj["name"] == "cold":
            from system.embedding.cold import ColdEmbeddingCache

            cache = ColdEmbeddingCache(root, keep_alive=1.0)
        else:
            raise RuntimeError("internal error")

        if eobj["index"] == "annoy":
            from system.embedding.annoy import AnnoyEmbeddingStore
            return AnnoyEmbeddingStore(
                namespace,
                providers,
                cache,
                root,
                trees=eobj["trees"],
                shard_size=eobj["shard_size"],
                is_dot=eobj["metric"] == "dot")
        raise ValueError(f"unsupported embedding index: {eobj['index']}")
    raise ValueError(f"unknown embed store: {eobj}")
