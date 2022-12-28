import os
from typing import Iterable

import torch
from annoy import AnnoyIndex

from misc.env import envload_path
from misc.io import ensure_folder
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
            embed_root: str) -> None:
        super().__init__(providers, cache)
        base_path = envload_path("USER_PATH", default="userdata")
        self._path = ensure_folder(os.path.join(base_path, embed_root))
        self._indexes: dict[ProviderRole, AnnoyIndex] = {}
        self._tmpindex: dict[ProviderRole, AnnoyIndex] = {}

    def _get_file(self, role: ProviderRole) -> str:
        provider = self.get_provider(role)
        return os.path.join(
            self._path, f"index.{provider.get_file_name()}.ann")

    def _create_index(self, role: ProviderRole, *, load: bool) -> AnnoyIndex:
        aindex = AnnoyIndex(self.num_dimensions(role), "dot")
        fname = self._get_file(role)
        if load and os.path.exists(fname):
            aindex.load(fname)
        return aindex

    def _get_index(self, role: ProviderRole) -> AnnoyIndex:
        res = self._indexes.get(role)
        if res is None:
            res = self._create_index(role, load=True)
            self._indexes[role] = res
        return res

    def do_index_init(
            self,
            role: ProviderRole) -> None:
        aindex = self._create_index(role, load=False)
        fname = self._get_file(role)
        aindex.on_disk_build(fname)
        self._tmpindex[role] = aindex

    def do_index_add(
            self,
            role: ProviderRole,
            index: int,
            embed: torch.Tensor) -> None:
        self._tmpindex[role].add_item(index, embed.ravel().tolist())

    def do_index_finish(self, role: ProviderRole) -> None:
        aindex: AnnoyIndex | None = self._tmpindex.pop(role, None)
        if aindex is None:
            raise RuntimeError("tmp index does not exist")
        aindex.build(100)
        self._indexes[role] = aindex

    def get_index_closest(
            self,
            role: ProviderRole,
            embed: torch.Tensor,
            count: int) -> Iterable[tuple[int, float]]:
        aindex = self._get_index(role)
        elems, dists = aindex.get_nns_by_vector(
            embed.ravel().tolist(), count, include_distances=True)
        yield from zip(elems, dists)

    @staticmethod
    def get_distance(embed_a: torch.Tensor, embed_b: torch.Tensor) -> float:
        return torch.dot(embed_a, embed_b).item()

    @staticmethod
    def is_bigger_better() -> bool:
        return True
