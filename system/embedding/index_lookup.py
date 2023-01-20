import math
import threading
from typing import Iterable, Literal

import torch

from model.embedding import (
    EmbeddingProvider,
    EmbeddingProviderMap,
    ProviderRole,
)
from system.embedding.processing import run_index_lookup
from system.embedding.store import EmbeddingStore
from system.msgs.message import MHash
from system.namespace.module import UnsupportedInit
from system.namespace.namespace import Namespace


LockState = Literal["free", "locked", "dead"]
LOCK_FREE: LockState = "free"
LOCK_LOCK: LockState = "locked"
LOCK_DEAD: LockState = "dead"


class EmbeddingCache:
    @staticmethod
    def cache_name() -> str:
        raise NotImplementedError()

    def is_cache_init(self) -> bool:
        return True

    def initialize_cache(self) -> None:
        raise UnsupportedInit(
            f"{self.cache_name()} cache does not support initialization!")

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

    def embedding_count(self, provider: EmbeddingProvider) -> int:
        raise NotImplementedError()

    def embeddings(
            self,
            provider: EmbeddingProvider,
            *,
            start_ix: int,
            limit: int | None,
            ) -> Iterable[tuple[int, MHash, torch.Tensor]]:
        raise NotImplementedError()


class CachedIndexEmbeddingStore(EmbeddingStore):
    def __init__(
            self,
            namespace: Namespace,
            providers: EmbeddingProviderMap,
            cache: EmbeddingCache,
            shard_size: int) -> None:
        super().__init__(providers)
        self._namespace = namespace
        self._cache = cache
        self._shard_size = shard_size

        self._request_build: set[ProviderRole] = set()
        self._lock = threading.RLock()
        self._cond = threading.Condition(lock=self._lock)
        self._th: threading.Thread | None = None
        self._err: BaseException | None = None

    def _start_build_loop(self) -> None:

        def build_loop() -> None:
            try:
                while True:
                    if th is not self._th:
                        break
                    while not self._request_build:
                        with self._lock:
                            self._cond.wait_for(
                                lambda: self._request_build, timeout=1.0)
                    while self._request_build:
                        elem = None
                        with self._lock:
                            rbl = list(self._request_build)
                            if rbl:
                                elem = rbl[0]
                                self._request_build.discard(elem)
                        if elem is not None:
                            self._build_index(elem)
            except BaseException as e:  # pylint: disable=broad-except
                self._err = e
            finally:
                with self._lock:
                    if th is self._th:
                        self._th = None

        self._check_err()
        with self._lock:
            if self._th is not None:
                return
            th = threading.Thread(target=build_loop, daemon=True)
            self._th = th
            th.start()

    def _check_err(self) -> None:
        if self._err is not None:
            raise self._err

    def maybe_request_build(self, role: ProviderRole) -> None:
        self._check_err()
        if role in self._request_build:
            return
        cache = self._cache
        provider = self.get_provider(role)
        shard = self.shard_of_index(cache.embedding_count(provider)) - 1
        if shard >= 0 and not self.is_shard_available(role, shard):
            with self._lock:
                self._request_build.add(role)
                self._cond.notify_all()
                self._start_build_loop()

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

    def get_all_embeddings(
            self,
            role: ProviderRole,
            *,
            progress_bar: bool) -> Iterable[tuple[MHash, torch.Tensor]]:
        cache = self._cache
        provider = self.get_provider(role)
        if not progress_bar:
            yield from (
                (mhash, embed)
                for _, mhash, embed in cache.embeddings(
                    provider, start_ix=0, limit=None))
            return
        # FIXME: add stubs
        from tqdm.auto import tqdm  # type: ignore

        count = cache.embedding_count(provider)
        with tqdm(total=count) as pbar:
            for _, mhash, embed in cache.embeddings(
                    provider, start_ix=0, limit=None):
                yield (mhash, embed)
                pbar.update(1)

    def proc_build_index_shard(self, role: ProviderRole, shard: int) -> None:
        cache = self._cache
        provider = self.get_provider(role)
        shard_size = self.shard_size()
        if self.is_shard_available(role, shard):
            return
        start_ix = self.get_shard_start_ix(shard)
        cur_embeds: list[tuple[MHash, torch.Tensor]] = []
        for _, mhash, embed in cache.embeddings(
                provider, start_ix=start_ix, limit=shard_size):
            cur_embeds.append((mhash, embed))
        if len(cur_embeds) == shard_size:
            self.do_build_index(
                role, shard, [embed for _, embed in cur_embeds])

    def set_index_lock_state(
            self, role: ProviderRole, shard: int, pid: int | None) -> None:
        raise NotImplementedError()

    def get_index_lock_state(
            self, role: ProviderRole, shard: int) -> LockState:
        raise NotImplementedError()

    def can_read_index(self, role: ProviderRole, shard: int) -> bool:
        return (
            self.is_shard_available(role, shard)
            and self.get_index_lock_state(role, shard) == LOCK_FREE)

    def can_build_index(self, role: ProviderRole, shard: int) -> bool:
        return self.get_index_lock_state(role, shard) != LOCK_LOCK

    def _build_index(self, role: ProviderRole) -> None:
        cache = self._cache
        provider = self.get_provider(role)
        total_count = cache.embedding_count(provider)
        shard_count = self.shard_of_index(total_count)

        def process(shard: int, line: str) -> None:
            if not line:
                return
            raise ValueError(f"did not expect output in shard {shard}: {line}")

        def on_err(err: BaseException) -> None:
            self._err = err

        threads = [
            threading.Thread(
                target=run_index_lookup,
                args=(
                    self._namespace,
                    role,
                    shard,
                    None,
                    0,
                    False,
                    process,
                    on_err))
            for shard in range(shard_count)
        ]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        self._check_err()

    def do_add_embedding(
            self,
            role: ProviderRole,
            mhash: MHash,
            embed: torch.Tensor) -> None:
        cache = self._cache
        provider = self.get_provider(role)
        cache.set_map_embedding(provider, mhash, embed)
        self.maybe_request_build(role)

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
        for ix_a, _, embed_a in cache.embeddings(
                provider, start_ix=0, limit=None):
            min_val = None
            max_val = None
            same_val = None
            line = 0
            for ix_b, _, embed_b in cache.embeddings(
                    provider, start_ix=0, limit=None):
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
        cache = self._cache
        provider = self.get_provider(role)
        candidates: dict[int, list[tuple[MHash, float]]] = {}
        total_count = cache.embedding_count(provider)
        shard_count = self.shard_of_index(total_count) + 1
        shard_size = self.shard_size()

        def process(shard: int, line: str) -> None:
            if not line:
                return
            left, right = line.split(",", 1)
            mhash = MHash.parse(left)
            dist = float(right)
            candidates[shard].append((mhash, dist))

        def on_err(err: BaseException) -> None:
            self._err = err

        threads = [
            threading.Thread(
                target=run_index_lookup,
                args=(
                    self._namespace,
                    role,
                    shard,
                    embed,
                    count,
                    precise,
                    process,
                    on_err))
            for shard in range(shard_count)
        ]
        for shard in range(shard_count):
            candidates[shard] = []
            end_ix = self.get_shard_start_ix(shard) + shard_size
            if (
                    not precise
                    and end_ix <= total_count
                    and not self.can_read_index(role, shard)):
                self.maybe_request_build(role)
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        self._check_err()
        yield from (
            sentry[0]
            for sentry in sorted(
                [
                    entry
                    for shard_candidates in candidates.values()
                    for entry in shard_candidates
                ],
                key=lambda entry: entry[1],
                reverse=self.is_bigger_better())[:count]
        )

    def proc_get_closest(
            self,
            role: ProviderRole,
            shard: int,
            embed: torch.Tensor,
            count: int,
            ignore_index: bool) -> Iterable[tuple[MHash, float]]:
        precise = ignore_index
        cache = self._cache
        provider = self.get_provider(role)
        start_ix = self.get_shard_start_ix(shard)

        if not precise and self.can_read_index(role, shard):
            yield from (
                (cache.get_entry_by_index(provider, start_ix + ix), dist)
                for ix, dist in self.get_index_closest(
                    role, shard, embed, count)
            )
            return
        shard_size = self.shard_size()
        yield from self._precise_closest(embed, count, (
            (mhash, other_embed)
            for _, mhash, other_embed in
            cache.embeddings(provider, start_ix=start_ix, limit=shard_size)))

    def _precise_closest(
            self,
            embed: torch.Tensor,
            count: int,
            embeddings: Iterable[tuple[MHash, torch.Tensor]],
            ) -> Iterable[tuple[MHash, float]]:
        is_bigger_better = self.is_bigger_better()

        def is_better(dist_new: float, dist_old: float) -> bool:
            if is_bigger_better:
                return dist_new > dist_old
            return dist_new < dist_old

        def is_already(mhash: MHash) -> bool:
            return mhash in (elem[0] for elem in candidates)

        candidates: list[tuple[MHash, float]] = []
        for mhash, other_embed in embeddings:
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
        yield from candidates
