from contextlib import contextmanager
from typing import Iterable, Iterator

import torch

from model.embedding import EmbeddingProvider
from system.embedding.store import EmbeddingStore
from system.msgs.message import MHash


REBUILD_THRESHOLD = 1000
CLOSEST_COUNT = 10


class EmbeddingCache:
    @contextmanager
    def get_lock(self) -> Iterator[None]:
        raise NotImplementedError()

    def set_embedding(
            self, name: str, mhash: MHash, embed: torch.Tensor) -> None:
        raise NotImplementedError()

    def get_embedding(self, name: str, mhash: MHash) -> torch.Tensor | None:
        raise NotImplementedError()

    def get_entry_by_index(self, name: str, index: int) -> MHash:
        raise NotImplementedError()

    def embedding_count(self, name: str) -> int:
        raise NotImplementedError()

    def embeddings(self, name: str) -> Iterable[tuple[int, torch.Tensor]]:
        raise NotImplementedError()

    def add_staging_embedding(
            self, name: str, mhash: MHash, embed: torch.Tensor) -> None:
        raise NotImplementedError()

    def staging_embeddings(
            self, name: str) -> Iterable[tuple[int, torch.Tensor]]:
        raise NotImplementedError()

    def get_staging_entry_by_index(self, name: str, index: int) -> MHash:
        raise NotImplementedError()

    def staging_offset(self, name: str) -> int:
        return self.embedding_count(name)

    def staging_count(self, name: str) -> int:
        raise NotImplementedError()

    def clear_staging(self) -> None:
        raise NotImplementedError()

    def all_embeddings(self, name: str) -> Iterable[tuple[int, torch.Tensor]]:
        yield from self.embeddings(name)
        offset = self.staging_offset(name)
        yield from (
            (offset + ix, embed)
            for (ix, embed) in self.staging_embeddings(name)
        )

    def get_by_index(self, name: str, index: int) -> MHash:
        offset = self.staging_offset(name)
        if index < offset:
            return self.get_entry_by_index(name, index)
        return self.get_staging_entry_by_index(name, index - offset)


class CachedIndexEmbeddingStore(EmbeddingStore):
    def __init__(
            self,
            providers: list[EmbeddingProvider],
            cache: EmbeddingCache) -> None:
        super().__init__(providers)
        self._cache = cache

    def do_build_index(
            self,
            name: str,
            num_dimensions: int,
            elements: Iterable[tuple[int, torch.Tensor]]) -> None:
        raise NotImplementedError()

    def get_index_closest(
            self,
            name: str,
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

    def build_index(self, name: str) -> None:
        cache = self._cache
        with cache.get_lock():
            provider = self.get_provider(name)
            self.do_build_index(
                name,
                provider.num_dimensions(),
                cache.all_embeddings(name))
            cache.clear_staging()

    def do_add_embedding(
            self,
            name: str,
            mhash: MHash,
            embed: torch.Tensor) -> None:
        cache = self._cache
        with cache.get_lock():
            cache.set_embedding(name, mhash, embed)
            cache.add_staging_embedding(name, mhash, embed)
            if cache.staging_count(name) > REBUILD_THRESHOLD:
                self.build_index(name)

    def do_get_embedding(
            self,
            name: str,
            mhash: MHash) -> torch.Tensor | None:
        return self._cache.get_embedding(name, mhash)

    def do_get_closest(
            self, name: str, embed: torch.Tensor) -> Iterable[MHash]:
        count = CLOSEST_COUNT
        cache = self._cache
        candidates = list(self.get_index_closest(name, embed, count))
        offset = cache.staging_offset(name)
        for other_ix, other_embed in cache.staging_embeddings(name):
            candidates.append(
                (other_ix + offset, self.get_distance(embed, other_embed)))
        yield from (
            cache.get_by_index(name, ix)
            for ix, _ in sorted(
                candidates,
                key=lambda entry: entry[1],
                reverse=self.is_bigger_better())[:count]
        )
