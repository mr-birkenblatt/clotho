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
    def get_lock(self, name: str) -> Iterator[None]:
        raise NotImplementedError()

    def set_map_embedding(
            self, name: str, mhash: MHash, embed: torch.Tensor) -> None:
        raise NotImplementedError()

    def get_map_embedding(
            self, name: str, mhash: MHash) -> torch.Tensor | None:
        raise NotImplementedError()

    def get_entry_by_index(self, name: str, index: int) -> MHash:
        raise NotImplementedError()

    def add_embedding(self, name: str, embed: torch.Tensor) -> int:
        raise NotImplementedError()

    def embedding_count(self, name: str) -> int:
        raise NotImplementedError()

    def embeddings(self, name: str) -> Iterable[tuple[int, torch.Tensor]]:
        raise NotImplementedError()

    def clear_embeddings(self, name: str) -> None:
        raise NotImplementedError()

    def add_staging_embedding(
            self, name: str, embed: torch.Tensor) -> None:
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

    def clear_staging(self, name: str) -> None:
        raise NotImplementedError()

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
        self._bulk = False

    def do_index_init(
            self,
            name: str) -> None:
        raise NotImplementedError()

    def do_index_add(
            self,
            name: str,
            index: int,
            embed: torch.Tensor) -> None:
        raise NotImplementedError()

    def do_index_finish(self, name: str) -> None:
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

    def num_dimensions(self, name: str) -> int:
        provider = self.get_provider(name)
        return provider.num_dimensions()

    def build_index(self, name: str) -> None:
        cache = self._cache
        with cache.get_lock(name):
            self.do_index_init(name)
            for index, embed in cache.embeddings(name):
                self.do_index_add(name, index, embed)
            for _, embed in cache.staging_embeddings(name):
                index = cache.add_embedding(name, embed)
                self.do_index_add(name, index, embed)
            self.do_index_finish(name)
            cache.clear_staging(name)

    @contextmanager
    def bulk_add(self, name: str) -> Iterator[None]:
        try:
            self._bulk = True
            with self._cache.get_lock(name):
                yield
                self.build_index(name)
        finally:
            self._bulk = False

    def do_add_embedding(
            self,
            name: str,
            mhash: MHash,
            embed: torch.Tensor) -> None:
        cache = self._cache
        with cache.get_lock(name):
            cache.set_map_embedding(name, mhash, embed)
            cache.add_staging_embedding(name, embed)
            if (not self._bulk
                    and cache.staging_count(name) > REBUILD_THRESHOLD):
                self.build_index(name)

    def do_get_embedding(
            self,
            name: str,
            mhash: MHash) -> torch.Tensor | None:
        return self._cache.get_map_embedding(name, mhash)

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
