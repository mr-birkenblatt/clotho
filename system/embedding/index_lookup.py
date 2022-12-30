from contextlib import contextmanager
from typing import Iterable, Iterator

import torch

from model.embedding import (
    EmbeddingProvider,
    EmbeddingProviderMap,
    ProviderRole,
)
from system.embedding.store import EmbeddingStore
from system.msgs.message import MHash
from system.msgs.store import MessageStore


REBUILD_THRESHOLD = 1000


class EmbeddingCache:
    @contextmanager
    def get_lock(self, provider: EmbeddingProvider) -> Iterator[None]:
        raise NotImplementedError()

    def set_map_embedding(
            self,
            provider: EmbeddingProvider,
            mhash: MHash,
            embed: torch.Tensor) -> None:
        raise NotImplementedError()

    def get_map_embedding(
            self,
            provider: EmbeddingProvider,
            mhash: MHash) -> torch.Tensor | None:
        raise NotImplementedError()

    def get_entry_by_index(
            self, provider: EmbeddingProvider, index: int) -> MHash:
        raise NotImplementedError()

    def add_embedding(self, provider: EmbeddingProvider, mhash: MHash) -> int:
        raise NotImplementedError()

    def embedding_count(self, provider: EmbeddingProvider) -> int:
        raise NotImplementedError()

    def embeddings(
            self,
            provider: EmbeddingProvider,
            ) -> Iterable[tuple[int, MHash, torch.Tensor]]:
        raise NotImplementedError()

    def clear_embeddings(self, provider: EmbeddingProvider) -> None:
        raise NotImplementedError()

    def add_staging_embedding(
            self, provider: EmbeddingProvider, mhash: MHash) -> int:
        raise NotImplementedError()

    def staging_embeddings(
            self,
            provider: EmbeddingProvider,
            ) -> Iterable[tuple[int, MHash, torch.Tensor]]:
        raise NotImplementedError()

    def get_staging_entry_by_index(
            self, provider: EmbeddingProvider, index: int) -> MHash:
        raise NotImplementedError()

    def staging_offset(self, provider: EmbeddingProvider) -> int:
        return self.embedding_count(provider)

    def staging_count(self, provider: EmbeddingProvider) -> int:
        raise NotImplementedError()

    def clear_staging(self, provider: EmbeddingProvider) -> None:
        raise NotImplementedError()

    def get_by_index(self, provider: EmbeddingProvider, index: int) -> MHash:
        offset = self.staging_offset(provider)
        if index < offset:
            return self.get_entry_by_index(provider, index)
        return self.get_staging_entry_by_index(provider, index - offset)

    def all_embeddings(
            self,
            provider: EmbeddingProvider,
            ) -> Iterable[tuple[int, MHash, torch.Tensor]]:
        yield from self.embeddings(provider)
        offset = self.staging_offset(provider)
        yield from (
            (index + offset, mhash, embed)
            for index, mhash, embed in self.staging_embeddings(provider)
        )


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
        provider = self.get_provider(role)
        with cache.get_lock(provider):
            self.do_index_init(role)
            for index, _, embed in cache.embeddings(provider):
                self.do_index_add(role, index, embed)
            for _, mhash, embed in cache.staging_embeddings(provider):
                index = cache.add_embedding(provider, mhash)
                self.do_index_add(role, index, embed)
            self.do_index_finish(role)
            cache.clear_staging(provider)

    def ensure_all(
            self,
            msg_store: MessageStore,
            roles: list[ProviderRole] | None = None) -> None:
        if roles is None:
            roles = self.get_roles()
        cache = self._cache

        def process_name(
                role: ProviderRole, provider: EmbeddingProvider) -> None:
            cache.clear_embeddings(provider)
            cache.clear_staging(provider)
            self.do_index_init(role)
            for index, mhash in enumerate(
                    msg_store.enumerate_messages(progress_bar=True)):
                embed = self.get_embedding(msg_store, role, mhash)
                self.do_index_add(role, index, embed)
            for _, mhash, embed in cache.staging_embeddings(provider):
                index = cache.add_embedding(provider, mhash)
                self.do_index_add(role, index, embed)
            self.do_index_finish(role)
            cache.clear_staging(provider)

        try:
            self._bulk = True
            for role in roles:
                provider = self.get_provider(role)
                with self._cache.get_lock(provider):
                    process_name(role, provider)
        finally:
            self._bulk = False

    @contextmanager
    def bulk_add(self, role: ProviderRole) -> Iterator[None]:
        try:
            self._bulk = True
            provider = self.get_provider(role)
            with self._cache.get_lock(provider):
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
        provider = self.get_provider(role)
        with cache.get_lock(provider):
            cache.set_map_embedding(provider, mhash, embed)
            cache.add_staging_embedding(provider, mhash)
            if (not self._bulk
                    and cache.staging_count(provider) > REBUILD_THRESHOLD):
                self.build_index(role)

    def do_get_embedding(
            self,
            role: ProviderRole,
            mhash: MHash) -> torch.Tensor | None:
        provider = self.get_provider(role)
        return self._cache.get_map_embedding(provider, mhash)

    def do_get_closest(
            self,
            role: ProviderRole,
            embed: torch.Tensor,
            count: int,
            *,
            precise: bool) -> Iterable[MHash]:
        if precise:
            yield from self.precise_closest(role, embed, count)
            return
        cache = self._cache
        provider = self.get_provider(role)
        candidates = list(self.get_index_closest(role, embed, count))
        offset = cache.staging_offset(provider)
        for other_ix, _, other_embed in cache.staging_embeddings(provider):
            candidates.append(
                (other_ix + offset, self.get_distance(embed, other_embed)))
        yield from (
            cache.get_by_index(provider, ix)
            for ix, _ in sorted(
                candidates,
                key=lambda entry: entry[1],
                reverse=self.is_bigger_better())[:count]
        )

    def precise_closest(
            self,
            role: ProviderRole,
            embed: torch.Tensor,
            count: int) -> Iterable[MHash]:
        cache = self._cache
        provider = self.get_provider(role)
        is_bigger_better = self.is_bigger_better()

        def is_better(dist_new: float, dist_old: float) -> bool:
            if is_bigger_better:
                return dist_new > dist_old
            return dist_new < dist_old

        def is_already(mhash: MHash) -> bool:
            return mhash in (elem[0] for elem in candidates)

        total = 0
        candidates: list[tuple[MHash, float]] = []
        for _, mhash, other_embed in cache.all_embeddings(provider):
            dist = self.get_distance(embed, other_embed)
            mod = False
            if len(candidates) < count:
                if not is_already(mhash):
                    candidates.append((mhash, dist))
                    mod = True
            elif is_better(dist, candidates[-1][1]):
                if not is_already(mhash):
                    candidates[-1] = (mhash, dist)
                    mod = True
            if mod:
                candidates.sort(
                    key=lambda entry: entry[1], reverse=is_bigger_better)
            total += 1
        print(f"checked {total} embeddings")
        yield from (mhash for mhash, _ in candidates)
