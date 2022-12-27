import os
from typing import Iterable

import torch
from annoy import AnnoyIndex

from misc.env import envload_path
from misc.io import ensure_folder
from model.embedding import EmbeddingProvider
from system.embedding.index_lookup import (
    CachedIndexEmbeddingStore,
    EmbeddingCache,
)


class AnnoyEmbeddingStore(CachedIndexEmbeddingStore):
    def __init__(
            self,
            providers: list[EmbeddingProvider],
            cache: EmbeddingCache,
            embed_root: str) -> None:
        super().__init__(providers, cache)
        base_path = envload_path("USER_PATH", default="userdata")
        self._path = ensure_folder(os.path.join(base_path, embed_root))
        self._indexes: dict[str, AnnoyIndex] = {}
        self._tmpindex: dict[str, AnnoyIndex] = {}

    def _get_file(self, name: str) -> str:
        return os.path.join(self._path, f"index_{name}.ann")

    def _create_index(self, name: str, *, load: bool) -> AnnoyIndex:
        aindex = AnnoyIndex(self.num_dimensions(name), "dot")
        fname = self._get_file(name)
        if load and os.path.exists(fname):
            aindex.load(fname)
        return aindex

    def _get_index(self, name: str) -> AnnoyIndex:
        res = self._indexes.get(name)
        if res is None:
            res = self._create_index(name, load=True)
            self._indexes = res
        return res

    def do_index_init(
            self,
            name: str) -> None:
        aindex = self._create_index(name, load=False)
        fname = self._get_file(name)
        aindex.on_disk_build(fname)
        self._tmpindex[name] = aindex

    def do_index_add(
            self,
            name: str,
            index: int,
            embed: torch.Tensor) -> None:
        self._tmpindex[name].add_item(index, embed.tolist())

    def do_index_finish(self, name: str) -> None:
        aindex: AnnoyIndex | None = self._tmpindex.pop(name, default=None)
        if aindex is None:
            raise RuntimeError("tmp index does not exist")
        aindex.build(100)
        self._indexes[name] = aindex

    def get_index_closest(
            self,
            name: str,
            embed: torch.Tensor,
            count: int) -> Iterable[tuple[int, float]]:
        aindex = self._get_index(name)
        elems, dists = aindex.get_nns_by_vector(
            embed.tolist(), count, include_distances=True)
        yield from zip(elems, dists)

    @staticmethod
    def get_distance(embed_a: torch.Tensor, embed_b: torch.Tensor) -> float:
        return torch.dot(embed_a, embed_b).item()

    @staticmethod
    def is_bigger_better() -> bool:
        return True
