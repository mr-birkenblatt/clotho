import contextlib
import math
from typing import Iterable, Iterator

import torch

from model.embedding import (
    EmbeddingProvider,
    EmbeddingProviderMap,
    PROVIDER_ROLES,
    ProviderRole,
)
from system.embedding.store import EmbeddingStore, get_embed_store
from system.msgs.message import MHash
from system.msgs.store import MessageStore
from system.namespace.module import UnsupportedInit, UnsupportedTransfer
from system.namespace.namespace import Namespace


class EmbeddingCache:
    @staticmethod
    def cache_name() -> str:
        raise NotImplementedError()

    def is_cache_init(self) -> bool:
        return True

    def initialize_cache(self) -> None:
        raise UnsupportedInit(
            f"{self.cache_name()} cache does not support initialization!")

    @contextlib.contextmanager
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

    def add_embedding(self, provider: EmbeddingProvider, mhash: MHash) -> None:
        raise NotImplementedError()

    def embedding_count(self, provider: EmbeddingProvider) -> int:
        raise NotImplementedError()

    def embeddings(
            self,
            provider: EmbeddingProvider,
            *,
            start_ix: int,
            ) -> Iterable[tuple[int, MHash, torch.Tensor]]:
        raise NotImplementedError()

    def clear_embeddings(self, provider: EmbeddingProvider) -> None:
        raise NotImplementedError()

    def add_staging_embedding(
            self, provider: EmbeddingProvider, mhash: MHash) -> None:
        raise NotImplementedError()

    def staging_embeddings(
            self,
            provider: EmbeddingProvider,
            *,
            remove: bool,
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
        yield from self.embeddings(provider, start_ix=0)
        offset = self.staging_offset(provider)
        yield from (
            (index + offset, mhash, embed)
            for index, mhash, embed in self.staging_embeddings(
                provider, remove=False)
        )


class CachedIndexEmbeddingStore(EmbeddingStore):
    def __init__(
            self,
            providers: EmbeddingProviderMap,
            cache: EmbeddingCache,
            shard_size: int) -> None:
        super().__init__(providers)
        self._cache = cache
        self._shard_size = shard_size
        self._bulk = False

    def shard_size(self) -> int:
        return self._shard_size

    def shard_of_index(self, index: int) -> int:
        return index // self._shard_size

    def index_in_shard(self, index: int) -> int:
        return index % self._shard_size

    def get_shard_start_ix(self, shard: int) -> int:
        return shard * self._shard_size

    def get_cache(self) -> EmbeddingCache:
        return self._cache

    def is_module_init(self) -> bool:
        return self._cache.is_cache_init()

    def initialize_module(self) -> None:
        return self._cache.initialize_cache()

    def from_namespace(
            self, other_namespace: Namespace, *, progress_bar: bool) -> None:
        oembed = get_embed_store(other_namespace)
        if not isinstance(oembed, CachedIndexEmbeddingStore):
            raise UnsupportedTransfer("nothing to transfer")
        ocache = oembed.get_cache()
        cache = self.get_cache()
        try:
            self._bulk = True
            for role in PROVIDER_ROLES:
                oprovider = oembed.get_provider(role)
                provider = self.get_provider(role)
                with ocache.get_lock(oprovider), cache.get_lock(provider):
                    for _, mhash, embed in ocache.all_embeddings(oprovider):
                        cache.set_map_embedding(provider, mhash, embed)
                        cache.add_staging_embedding(provider, mhash)
        finally:
            self._bulk = False

    def do_build_index(
            self,
            role: ProviderRole,
            shard: int,
            embeds: list[torch.Tensor]) -> None:
        raise NotImplementedError()

    def get_index_closest(
            self,
            role: ProviderRole,
            shard: int,
            embed: torch.Tensor,
            count: int) -> Iterable[tuple[int, float]]:
        raise NotImplementedError()

    def is_shard_available(self, role: ProviderRole, shard: int) -> bool:
        raise NotImplementedError()

    # FIXME: could be bulk operation
    def get_distance(
            self, embed_a: torch.Tensor, embed_b: torch.Tensor) -> float:
        raise NotImplementedError()

    def is_bigger_better(self) -> bool:
        raise NotImplementedError()

    def num_dimensions(self, role: ProviderRole) -> int:
        return self.get_provider(role).num_dimensions()

    def build_index(self, role: ProviderRole) -> None:
        cache = self._cache
        provider = self.get_provider(role)
        shard_size = self.shard_size()
        with cache.get_lock(provider):
            shard = 0
            while self.is_shard_available(role, shard):
                shard += 1
            start_ix = self.get_shard_start_ix(shard)
            cur_embeds: list[tuple[MHash, torch.Tensor]] = []
            for _, mhash, embed in cache.embeddings(
                    provider, start_ix=start_ix):
                cur_embeds.append((mhash, embed))
                if len(cur_embeds) == shard_size:
                    self.do_build_index(
                        role, shard, [embed for _, embed in cur_embeds])
                    shard += 1
                    cur_embeds = []
            try:
                for _, mhash, embed in cache.staging_embeddings(
                        provider, remove=True):
                    cache.add_embedding(provider, mhash)
                    cur_embeds.append((mhash, embed))
                    if len(cur_embeds) == shard_size:
                        self.do_build_index(
                            role, shard, [embed for _, embed in cur_embeds])
                        shard += 1
                        cur_embeds = []
            finally:
                for mhash, _ in cur_embeds:
                    cache.add_staging_embedding(provider, mhash)

    def ensure_all(
            self,
            msg_store: MessageStore,
            roles: list[ProviderRole] | None = None) -> None:
        if roles is None:
            roles = self.get_roles()
        cache = self._cache
        shard_size = self.shard_size()

        def process_name(
                role: ProviderRole, provider: EmbeddingProvider) -> None:
            shard = 0
            cur_embeds: list[tuple[MHash, torch.Tensor]] = []
            for mhash in msg_store.enumerate_messages(progress_bar=True):
                cache.add_embedding(provider, mhash)
                embed = self.get_embedding(msg_store, role, mhash)
                cur_embeds.append((mhash, embed))
                if len(cur_embeds) == shard_size:
                    self.do_build_index(
                        role, shard, [embed for _, embed in cur_embeds])
                    shard += 1
                    cur_embeds = []
            try:
                for _, mhash, embed in cache.staging_embeddings(
                        provider, remove=True):
                    cache.add_embedding(provider, mhash)
                    cur_embeds.append((mhash, embed))
                    if len(cur_embeds) == shard_size:
                        self.do_build_index(
                            role, shard, [embed for _, embed in cur_embeds])
                        shard += 1
                        cur_embeds = []
            finally:
                for mhash, _ in cur_embeds:
                    cache.add_staging_embedding(provider, mhash)

        try:
            self._bulk = True
            for role in roles:
                provider = self.get_provider(role)
                with self._cache.get_lock(provider):
                    process_name(role, provider)
        finally:
            self._bulk = False

    @contextlib.contextmanager
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
                    and cache.staging_count(provider) >= self.shard_size()):
                self.build_index(role)

    def do_get_embedding(
            self,
            role: ProviderRole,
            mhash: MHash) -> torch.Tensor | None:
        provider = self.get_provider(role)
        return self._cache.get_map_embedding(provider, mhash)

    def do_get_internal_distance(
            self,
            role: ProviderRole,
            shard: int,
            index_a: int,
            index_b: int) -> float:
        raise NotImplementedError()

    def self_test(self, role: ProviderRole, count: int | None) -> None:
        cache = self._cache
        provider = self.get_provider(role)
        is_bigger_better = self.is_bigger_better()
        any_compute = False
        for ix_a, _, embed_a in cache.embeddings(provider, start_ix=0):
            min_val = None
            max_val = None
            same_val = None
            line = 0
            for ix_b, _, embed_b in cache.embeddings(provider, start_ix=0):
                external = self.get_distance(embed_a, embed_b)
                # print(
                #     ix_a,
                #     ix_b,
                #     external,
                #     embed_a.ravel()[:3],
                #     embed_b.ravel()[:3])
                if ix_a == ix_b:
                    same_val = external
                else:
                    if min_val is None or min_val > external:
                        min_val = external
                    if max_val is None or max_val < external:
                        max_val = external
                shard = 0
                while self.is_shard_available(role, shard):
                    internal = self.do_get_internal_distance(
                        role, shard, ix_a, ix_b)
                    if not math.isclose(internal, external, abs_tol=1e-3):
                        raise ValueError(
                            "distances are not equal: "
                            f"{internal} != {external} "
                            f"ix: {ix_a} {ix_b} embed: {embed_a} {embed_b}")
                    shard += 1
                any_compute = True
                line += 1
                if count is not None and ix_b >= count:
                    break
            if (min_val is not None
                    and max_val is not None
                    and same_val is not None):
                text = f"{min_val} {same_val} {max_val} ({line} values {ix_a})"
                if is_bigger_better:
                    if (math.isclose(min_val, same_val)
                            or min_val > same_val):
                        print(f"min is bigger than same for bb {text}")
                    if (not math.isclose(max_val, same_val)
                            and max_val > same_val):
                        print(f"max is bigger than same for bb {text}")
                else:
                    if (not math.isclose(min_val, same_val)
                            and min_val < same_val):
                        print(f"min is smaller than same for not bb {text}")
                    if (math.isclose(max_val, same_val)
                            or max_val < same_val):
                        print(f"max is smaller than same for not bb {text}")
            else:
                raise ValueError("not enough values")
            if count is not None and ix_a >= count:
                break
        if not any_compute:
            raise ValueError("no computation has happened")

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
        candidates: list[tuple[int, float]] = []
        shard = 0
        while self.is_shard_available(role, shard):
            start_ix = self.get_shard_start_ix(shard)
            candidates.extend((
                (start_ix + ix, dist)
                for ix, dist in self.get_index_closest(
                    role, shard, embed, count + count // 2)
            ))
            shard += 1
        last_shard_offset = self.get_shard_start_ix(shard)
        for last_ix, _, other_embed in cache.embeddings(
                provider, start_ix=last_shard_offset):
            candidates.append((last_ix, self.get_distance(embed, other_embed)))
        offset = cache.staging_offset(provider)
        for other_ix, _, other_embed in cache.staging_embeddings(
                provider, remove=False):
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
