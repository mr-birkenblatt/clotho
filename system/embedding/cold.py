import os
from typing import Iterable

import torch

from misc.cold_writer import ColdAccess
from model.embedding import EmbeddingProvider, PROVIDER_ROLES
from system.embedding.index_lookup import EmbeddingCache
from system.embedding.processing import (
    deserialize_embedding,
    serialize_embedding,
)
from system.msgs.message import MHash


class ColdEmbeddingCache(EmbeddingCache):
    def __init__(self, root: str, *, keep_alive: float) -> None:
        super().__init__()
        self._embeds = {
            role: ColdAccess(
                os.path.join(root, f"{role}.zip"), keep_alive=keep_alive)
            for role in PROVIDER_ROLES
        }

    @staticmethod
    def cache_name() -> str:
        return "cold"

    def set_map_embedding(
            self,
            provider: EmbeddingProvider,
            mhash: MHash,
            embed: torch.Tensor) -> None:
        self._embeds[provider.get_role()].write_line(
            f"{mhash.to_parseable()},{serialize_embedding(embed)}")

    def get_map_embedding(
            self,
            provider: EmbeddingProvider,
            mhash: MHash) -> torch.Tensor | None:
        raise RuntimeError("cannot random access from cold storage")

    def get_entry_by_index(
            self, provider: EmbeddingProvider, index: int) -> MHash:
        raise RuntimeError("cannot random access from cold storage")

    def embedding_count(self, provider: EmbeddingProvider) -> int:
        total = 0
        for line in self._embeds[provider.get_role()].enumerate_lines():
            if not line:
                continue
            total += 1
        return total

    def embeddings(
            self,
            provider: EmbeddingProvider,
            *,
            start_ix: int,
            limit: int | None,
            ) -> Iterable[tuple[int, MHash, torch.Tensor]]:
        for ix, line in enumerate(
                self._embeds[provider.get_role()].enumerate_lines()):
            if not line:
                continue
            if ix < start_ix:
                continue
            if limit is not None and ix >= start_ix + limit:
                break
            mhash, embed = line.split(",", 1)
            yield ix, MHash.parse(mhash), deserialize_embedding(embed)
