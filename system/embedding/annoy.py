import os
from typing import Iterable

import numpy as np
import torch
from annoy import AnnoyIndex

from misc.io import named_write, open_read, open_write, remove_file
from misc.util import check_pid_exists, ideal_thread_count, safe_ravel
from model.embedding import EmbeddingProviderMap, ProviderRole
from system.embedding.index_lookup import (
    CachedIndexEmbeddingStore,
    EmbeddingCache,
    LOCK_DEAD,
    LOCK_FREE,
    LOCK_LOCK,
    LockState,
)
from system.namespace.namespace import Namespace


class AnnoyEmbeddingStore(CachedIndexEmbeddingStore):
    def __init__(
            self,
            namespace: Namespace,
            providers: EmbeddingProviderMap,
            cache: EmbeddingCache,
            embed_root: str,
            *,
            trees: int,
            shard_size: int,
            is_dot: bool) -> None:
        super().__init__(namespace, providers, cache, shard_size)
        self._path = embed_root
        self._trees = trees
        self._annoy_cache: dict[tuple[ProviderRole, int], AnnoyIndex] = {}
        self._is_dot = is_dot

    def _get_file(self, role: ProviderRole, shard: int) -> str:
        provider = self.get_provider(role)
        return os.path.join(
            self._path, f"index.{provider.get_file_name()}.{shard}.ann")

    def _get_lock_file(self, role: ProviderRole, shard: int) -> str:
        provider = self.get_provider(role)
        return os.path.join(
            self._path, f"lock.{provider.get_file_name()}.{shard}.ann")

    def set_index_lock_state(
            self, role: ProviderRole, shard: int, pid: int | None) -> None:
        # NOTE: this lock doesn't have to be precise
        # it is fine if multiple processes create the same index concurrently
        # since each uses their own tmpfile
        fname = self._get_lock_file(role, shard)
        if pid is None:
            remove_file(fname)
        else:
            with open_write(fname, text=True) as fout:
                fout.write(f"{pid}\n")

    def get_index_lock_state(
            self, role: ProviderRole, shard: int) -> LockState:
        fname = self._get_lock_file(role, shard)
        try:
            with open_read(fname, text=True) as fin:
                pid = int(fin.read().strip())
            return LOCK_LOCK if check_pid_exists(pid) else LOCK_DEAD
        except ValueError:
            return LOCK_DEAD
        except FileNotFoundError:
            return LOCK_FREE

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
        key = (role, shard)
        res = self._annoy_cache.get(key)
        if res is None:
            res = self._create_index(role, shard, load=True)
            self._annoy_cache[key] = res
        return res

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
            aindex.build(self._trees, n_jobs=max(1, ideal_thread_count() // 4))
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
