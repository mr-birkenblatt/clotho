from contextlib import contextmanager
from typing import Iterable, Iterator

# FIXME add stubs
import torch  # type: ignore

from model.embedding import (
    EmbeddingProvider,
    EmbeddingProviderMap,
    ProviderRole,
)
from system.embedding.store import EmbeddingStore
from system.msgs.message import MHash
from system.msgs.store import MessageStore


REBUILD_THRESHOLD = 1000
CLOSEST_COUNT = 10


class EmbeddingCache:
    @contextmanager
    def get_lock(self, provider: EmbeddingProvider) -> Iterator[None]:
        raise NotImplementedError()

    def set_map_embedding(
            self, provider: EmbeddingProvider, mhash: MHash, embed: torch.Tensor) -> None:
        raise NotImplementedError()

    def get_map_embedding(
            self, provider: EmbeddingProvider, mhash: MHash) -> torch.Tensor | None:
        raise NotImplementedError()

    def get_entry_by_index(self, provider: EmbeddingProvider, index: int) -> MHash:
        raise NotImplementedError()

    def add_embedding(self, provider: EmbeddingProvider, mhash: MHash) -> int:
        raise NotImplementedError()

    def embedding_count(self, provider: EmbeddingProvider) -> int:
        raise NotImplementedError()

    def embeddings(
            self, provider: EmbeddingProvider) -> Iterable[tuple[int, MHash, torch.Tensor]]:
        raise NotImplementedError()

    def clear_embeddings(self, provider: EmbeddingProvider) -> None:
        raise NotImplementedError()

    def add_staging_embedding(self, provider: EmbeddingProvider, mhash: MHash) -> None:
        raise NotImplementedError()

    def staging_embeddings(
            self, provider: EmbeddingProvider) -> Iterable[tuple[int, MHash, torch.Tensor]]:
        raise NotImplementedError()

    def get_staging_entry_by_index(self, provider: EmbeddingProvider, index: int) -> MHash:
        raise NotImplementedError()

    def staging_offset(self, provider: EmbeddingProvider) -> int:
        return self.embedding_count(name)

    def staging_count(self, provider: EmbeddingProvider) -> int:
        raise NotImplementedError()

    def clear_staging(self, provider: EmbeddingProvider) -> None:
        raise NotImplementedError()

    def get_by_index(self, provider: EmbeddingProvider, index: int) -> MHash:
        offset = self.staging_offset(name)
        if index < offset:
            return self.get_entry_by_index(name, index)
        return self.get_staging_entry_by_index(name, index - offset)


class CachedIndexEmbeddingStore(EmbeddingStore):
    def __init__(
            self,
            providers: EmbeddingProviderMap,
            cache: EmbeddingCache) -> None:
        super().__init__(providers)
        self._cache = cache
        self._bulk = False

    def do_index_init(
            self,
            role: ProviderRole) -> None:
        raise NotImplementedError()

    def do_index_add(
            self,
            role: ProviderRole,
            index: int,
            embed: torch.Tensor) -> None:
        raise NotImplementedError()

    def do_index_finish(self, role: ProviderRole) -> None:
        raise NotImplementedError()

    def get_index_closest(
            self,
            role: ProviderRole,
            embed: torch.Tensor,
            count: int) -> Iterable[tuple[int, float]]:
        raise NotImplementedError()

    # FIXME: could be bulk operation
    @staticmethod
    def get_distance(embed_a: torch.Tensor, embed_b: torch.Tensor) -> float:
        raise NotImplementedError()

    @staticmethod
    def is_bigger_better() -> bool:
        raise NotImplementedError()

    def num_dimensions(self, role: ProviderRole) -> int:
        return self.get_provider(role).num_dimensions()

    def build_index(self, role: ProviderRole) -> None:
        cache = self._cache
        with cache.get_lock(role):
            self.do_index_init(role)
            for index, _, embed in cache.embeddings(role):
                self.do_index_add(role, index, embed)
            for _, mhash, embed in cache.staging_embeddings(role):
                index = cache.add_embedding(role, mhash)
                self.do_index_add(role, index, embed)
            self.do_index_finish(role)
            cache.clear_staging(role)

    def ensure_all(
            self,
            msg_store: MessageStore,
            roles: list[ProviderRole] | None = None) -> None:
        if roles is None:
            roles = self.get_roles()
        cache = self._cache

        def process_name(role: ProviderRole) -> None:
            self.do_index_init(role)
            for index, mhash in enumerate(
                    msg_store.enumerate_messages(progress_bar=True)):
                embed = self.get_embedding(msg_store, role, mhash)
                self.do_index_add(role, index, embed)
            for _, mhash, embed in cache.staging_embeddings(role):
                index = cache.add_embedding(role, mhash)
                self.do_index_add(role, index, embed)
            self.do_index_finish(role)
            cache.clear_staging(role)

        try:
            self._bulk = True
            for role in roles:
                with self._cache.get_lock(role):
                    process_name(role)
        finally:
            self._bulk = False

    @contextmanager
    def bulk_add(self, role: ProviderRole) -> Iterator[None]:
        try:
            self._bulk = True
            with self._cache.get_lock(role):
                yield
                self.build_index(role)
        finally:
            self._bulk = False

    def do_add_embedding(
            self,
            role: ProviderRole,
            mhash: MHash,
            embed: torch.Tensor) -> None:
        cache = self._cache
        with cache.get_lock(name):
            cache.set_map_embedding(name, mhash, embed)
            cache.add_staging_embedding(name, mhash)
            if (not self._bulk
                    and cache.staging_count(name) > REBUILD_THRESHOLD):
                self.build_index(name)

    def do_get_embedding(
            self,
            role: ProviderRole,
            mhash: MHash) -> torch.Tensor | None:
        return self._cache.get_map_embedding(name, mhash)

    def do_get_closest(
            self, role: ProviderRole, embed: torch.Tensor) -> Iterable[MHash]:
        count = CLOSEST_COUNT
        cache = self._cache
        candidates = list(self.get_index_closest(name, embed, count))
        offset = cache.staging_offset(name)
        for other_ix, _, other_embed in cache.staging_embeddings(name):
            candidates.append(
                (other_ix + offset, self.get_distance(embed, other_embed)))
        yield from (
            cache.get_by_index(name, ix)
            for ix, _ in sorted(
                candidates,
                key=lambda entry: entry[1],
                reverse=self.is_bigger_better())[:count]
        )
