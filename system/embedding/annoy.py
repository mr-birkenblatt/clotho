import os
from typing import Iterable

import numpy as np
import torch
from annoy import AnnoyIndex

from misc.env import envload_path
from misc.io import ensure_folder, fastrename, remove_file
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
            trees: int,
            is_dot: bool) -> None:
        super().__init__(providers, cache)
        base_path = envload_path("USER_PATH", default="userdata")
        self._path = ensure_folder(os.path.join(base_path, embed_root))
        self._indexes: dict[ProviderRole, AnnoyIndex] = {}
        self._tmpindex: dict[ProviderRole, AnnoyIndex] = {}
        self._trees = trees
        self._is_dot = is_dot

    def _get_file(self, role: ProviderRole, *, is_tmp: bool) -> str:
        provider = self.get_provider(role)
        tmp = ".~tmp" if is_tmp else ""
        return os.path.join(
            self._path, f"index.{provider.get_file_name()}.ann{tmp}")

    def _create_index(self, role: ProviderRole, *, load: bool) -> AnnoyIndex:
        aindex = AnnoyIndex(
            self.num_dimensions(role), "dot" if self._is_dot else "angular")
        if load:
            fname = self._get_file(role, is_tmp=False)
            tname = self._get_file(role, is_tmp=True)
            if os.path.exists(fname):
                aindex.load(fname)
            elif os.path.exists(tname):
                fastrename(tname, fname)
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
        fname = self._get_file(role, is_tmp=True)
        remove_file(fname)
        aindex.on_disk_build(fname)
        self._tmpindex[role] = aindex

    def do_index_add(
            self,
            role: ProviderRole,
            index: int,
            embed: torch.Tensor) -> None:
        self._tmpindex[role].add_item(index, safe_ravel(embed).tolist())

    def do_index_finish(self, role: ProviderRole) -> None:
        aindex: AnnoyIndex | None = self._tmpindex.pop(role, None)
        if aindex is None:
            raise RuntimeError("tmp index does not exist")
        aindex.build(self._trees)
        self._indexes[role] = aindex

    def do_get_internal_distance(
            self, role: ProviderRole, index_a: int, index_b: int) -> float:
        aindex = self._get_index(role)
        return aindex.get_distance(index_a, index_b)

    def get_index_closest(
            self,
            role: ProviderRole,
            embed: torch.Tensor,
            count: int) -> Iterable[tuple[int, float]]:
        aindex = self._get_index(role)
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
