import os
from typing import Iterable

import numpy as np
import torch
from annoy import AnnoyIndex

from misc.io import named_write
from misc.util import safe_ravel
from model.embedding import EmbeddingProviderMap, ProviderRole
from system.embedding.index_lookup import (
    CachedIndexEmbeddingStore,
    EmbeddingCache,
)


class AnnoyEmbeddingStore(CachedIndexEmbeddingStore):
    def __init__(
            self,
            providers: EmbeddingProviderMap,
            cache: EmbeddingCache,
            embed_root: str,
            *,
            trees: int,
            shard_size: int,
            is_dot: bool) -> None:
        super().__init__(providers, cache, shard_size)
        self._path = embed_root
        self._trees = trees
        self._is_dot = is_dot

    def _get_file(self, role: ProviderRole, shard: int) -> str:
        provider = self.get_provider(role)
        return os.path.join(
            self._path, f"index.{provider.get_file_name()}.{shard}.ann")

    def _create_index(
            self,
            role: ProviderRole,
            shard: int,
            *,
            load: bool) -> AnnoyIndex:
        aindex = AnnoyIndex(
            self.num_dimensions(role), "dot" if self._is_dot else "angular")
        if load:
            fname = self._get_file(role, shard)
            if os.path.exists(fname):
                aindex.load(fname)
        return aindex

    def is_shard_available(self, role: ProviderRole, shard: int) -> bool:
        fname = self._get_file(role, shard)
        if os.path.exists(fname):
            return True
        return False

    def _get_index(self, role: ProviderRole, shard: int) -> AnnoyIndex:
        return self._create_index(role, shard, load=True)

    def do_build_index(
            self,
            role: ProviderRole,
            shard: int,
            embeds: list[torch.Tensor]) -> None:
        with named_write(self._get_file(role, shard)) as tname:
            aindex = self._create_index(role, shard, load=False)
            aindex.on_disk_build(tname)
            for ix, embed in enumerate(embeds):
                aindex.add_item(ix, safe_ravel(embed).tolist())
            aindex.build(self._trees)
            aindex.unload()

    def do_get_internal_distance(
            self,
            role: ProviderRole,
            shard: int,
            index_a: int,
            index_b: int) -> float:
        aindex = self._get_index(role, shard)
        return aindex.get_distance(index_a, index_b)

    def get_index_closest(
            self,
            role: ProviderRole,
            shard: int,
            embed: torch.Tensor,
            count: int) -> Iterable[tuple[int, float]]:
        aindex = self._get_index(role, shard)
        elems, dists = aindex.get_nns_by_vector(
            safe_ravel(embed).tolist(), count, include_distances=True)
        yield from zip(elems, dists)

    def get_distance(
            self, embed_a: torch.Tensor, embed_b: torch.Tensor) -> float:
        if self._is_dot:
            return torch.dot(safe_ravel(embed_a), safe_ravel(embed_b)).item()
        cos = torch.nn.functional.cosine_similarity(
            safe_ravel(embed_a), safe_ravel(embed_b), dim=0).item()
        return np.sqrt(max(0.0, 2.0 - 2.0 * cos))

    def is_bigger_better(self) -> bool:
        return self._is_dot
