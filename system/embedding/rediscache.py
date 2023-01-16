import gzip
import io
from contextlib import contextmanager
from typing import Iterable, Iterator

import numpy as np
import torch

from misc.redis import ConfigKey, RedisConnection
from model.embedding import EmbeddingProvider
from system.embedding.index_lookup import EmbeddingCache
from system.msgs.message import MHash


class RedisEmbeddingCache(EmbeddingCache):
    def __init__(self, ns_key: ConfigKey) -> None:
        super().__init__()
        self._redis = RedisConnection(ns_key, "embed")

    @staticmethod
    def cache_name() -> str:
        return "redis"

    @contextmanager
    def get_lock(self, provider: EmbeddingProvider) -> Iterator[None]:
        name = provider.get_redis_name()
        with self._redis.get_lock(f"lock:{name}"):
            yield

    def _get_embedding_key(
            self, provider: EmbeddingProvider, mhash: MHash) -> str:
        name = provider.get_redis_name()
        return f"{self._redis.get_prefix()}:map:{name}:{mhash.to_parseable()}"

    def _get_staging_key(self, provider: EmbeddingProvider) -> str:
        name = provider.get_redis_name()
        return f"{self._redis.get_prefix()}:staging:{name}"

    def _get_order_key(self, provider: EmbeddingProvider) -> str:
        name = provider.get_redis_name()
        return f"{self._redis.get_prefix()}:order:{name}"

    def _serialize(self, embed: torch.Tensor) -> bytes:
        bout = io.BytesIO()
        with gzip.GzipFile(fileobj=bout, mode="w") as fout:
            np.save(fout, embed.detach().numpy())
        return bout.getvalue()

    def _deserialize(self, content: bytes) -> torch.Tensor:
        binp = io.BytesIO(content)
        with gzip.GzipFile(fileobj=binp, mode="r") as finp:
            return torch.Tensor(np.load(finp))

    def set_map_embedding(
            self,
            provider: EmbeddingProvider,
            mhash: MHash,
            embed: torch.Tensor) -> None:
        key = self._get_embedding_key(provider, mhash)
        with self._redis.get_connection(depth=0) as conn:
            conn.set(key, self._serialize(embed))

    def get_map_embedding(
            self,
            provider: EmbeddingProvider,
            mhash: MHash) -> torch.Tensor | None:
        key = self._get_embedding_key(provider, mhash)
        with self._redis.get_connection(depth=0) as conn:
            res = conn.get(key)
        if res is None:
            return None
        return self._deserialize(res)

    def get_entry_by_index(
            self, provider: EmbeddingProvider, index: int) -> MHash:
        key = self._get_order_key(provider)
        return self._get_index(key, index)

    def add_embedding(self, provider: EmbeddingProvider, mhash: MHash) -> int:
        key = self._get_order_key(provider)
        return self._add_embedding(key, mhash)

    def embedding_count(self, provider: EmbeddingProvider) -> int:
        key = self._get_order_key(provider)
        return self._embeddings_size(key)

    def embeddings(
            self,
            provider: EmbeddingProvider,
            ) -> Iterable[tuple[int, MHash, torch.Tensor]]:
        key = self._get_order_key(provider)
        return self._get_embeddigs(key, provider)

    def clear_embeddings(self, provider: EmbeddingProvider) -> None:
        key = self._get_order_key(provider)
        self._clear_embeddings(key)

    def add_staging_embedding(
            self, provider: EmbeddingProvider, mhash: MHash) -> int:
        key = self._get_staging_key(provider)
        return self._add_embedding(key, mhash)

    def staging_embeddings(
            self,
            provider: EmbeddingProvider,
            ) -> Iterable[tuple[int, MHash, torch.Tensor]]:
        key = self._get_staging_key(provider)
        return self._get_embeddigs(key, provider)

    def get_staging_entry_by_index(
            self, provider: EmbeddingProvider, index: int) -> MHash:
        key = self._get_staging_key(provider)
        return self._get_index(key, index)

    def staging_count(self, provider: EmbeddingProvider) -> int:
        key = self._get_staging_key(provider)
        return self._embeddings_size(key)

    def clear_staging(self, provider: EmbeddingProvider) -> None:
        key = self._get_staging_key(provider)
        self._clear_embeddings(key)

    def _add_embedding(self, key: str, mhash: MHash) -> int:
        with self._redis.get_connection(depth=0) as conn:
            res = int(conn.rpush(key, mhash.to_parseable().encode("utf-8")))
            return res - 1

    def _get_index(self, key: str, index: int) -> MHash:
        with self._redis.get_connection(depth=1) as conn:
            res = conn.lindex(key, index)
            if res is None:
                raise KeyError(f"index not in list: {key} {index}")
            return MHash.parse(res.decode("utf-8"))

    def _get_embeddigs(
            self,
            key: str,
            provider: EmbeddingProvider,
            ) -> Iterable[tuple[int, MHash, torch.Tensor]]:
        offset = 0
        batch_size = 100

        def as_mhash(elem: bytes) -> MHash:
            return MHash.parse(elem.decode("utf-8"))

        def as_tensor(mhash: MHash) -> torch.Tensor:
            tres = self.get_map_embedding(provider, mhash)
            if tres is None:
                raise KeyError(f"missing key: {mhash}")
            return tres

        def as_tuple(
                offset: int,
                ix: int,
                elem: bytes) -> tuple[int, MHash, torch.Tensor]:
            mhash = as_mhash(elem)
            return (offset + ix, mhash, as_tensor(mhash))

        with self._redis.get_connection(depth=1) as conn:
            while True:
                res = conn.lrange(key, offset, offset + batch_size)
                if not res:
                    break
                yield from (
                    as_tuple(offset, ix, elem)
                    for ix, elem in enumerate(res)
                )
                offset += batch_size

    def _embeddings_size(self, key: str) -> int:
        with self._redis.get_connection(depth=1) as conn:
            return int(conn.llen(key))

    def _clear_embeddings(self, key: str) -> None:
        with self._redis.get_connection(depth=1) as conn:
            conn.delete(key)
