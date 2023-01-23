import os
from typing import Iterable

import torch

from misc.cold_writer import ColdAccess
from model.embedding import EmbeddingProvider
from system.embedding.index_lookup import EmbeddingCache
from system.embedding.processing import (
    deserialize_embedding,
    serialize_embedding,
)
from system.msgs.message import MHash


class ColdEmbeddingCache(EmbeddingCache):
    def __init__(self, root: str, *, keep_alive: float) -> None:
        super().__init__()
        self._root = root
        self._keep_alive = keep_alive
        self._colds: list[ColdAccess] = []
        self._embeds: dict[str, int] = {}

    @staticmethod
    def cache_name() -> str:
        return "cold"

    def _get_basename(self, provider: EmbeddingProvider) -> str:
        return (
            f"{provider.get_role()}-"
            f"{provider.get_embedding_name()}-"
            f"v{provider.get_embedding_version()}-"
            f"{provider.get_embedding_hash()}"
            ".zip")

    def get_embedding_id_for(self, provider: EmbeddingProvider) -> int:
        base = self._get_basename(provider)
        res = self._embeds.get(base)
        if res is None:
            res = len(self._colds)
            fname = os.path.join(self._root, base)
            self._colds.append(ColdAccess(fname, keep_alive=self._keep_alive))
            self._embeds[base] = res
        return res

    def set_map_embedding(
            self,
            embedding_id: int,
            mhash: MHash,
            embed: torch.Tensor) -> None:
        self._colds[embedding_id].write_line(
            f"{mhash.to_parseable()},{serialize_embedding(embed)}")

    def get_map_embedding(
            self, embedding_id: int, mhash: MHash) -> torch.Tensor | None:
        raise RuntimeError("cannot random access from cold storage")

    def get_entry_by_index(self, embedding_id: int, *, index: int) -> MHash:
        raise RuntimeError("cannot random access from cold storage")

    def embedding_count(self, embedding_id: int) -> int:
        total = 0
        for line in self._colds[embedding_id].enumerate_lines():
            if not line:
                continue
            total += 1
        return total

    def embeddings(
            self,
            embedding_id: int,
            *,
            start_ix: int,
            limit: int | None,
            ) -> Iterable[tuple[int, MHash, torch.Tensor]]:
        for ix, line in enumerate(
                self._colds[embedding_id].enumerate_lines()):
            if not line:
                continue
            if ix < start_ix:
                continue
            if limit is not None and ix >= start_ix + limit:
                break
            mhash, embed = line.split(",", 1)
            yield ix, MHash.parse(mhash), deserialize_embedding(embed)
