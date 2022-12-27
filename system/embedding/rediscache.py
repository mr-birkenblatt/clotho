import pickle
from contextlib import contextmanager
from typing import Iterable, Iterator

import torch

from misc.redis import ConfigKey, RedisConnection
from system.embedding.index_lookup import EmbeddingCache
from system.msgs.message import MHash


class RedisEmbeddingCache(EmbeddingCache):
    def __init__(self, ns_key: ConfigKey) -> None:
        super().__init__()
        self._redis = RedisConnection(ns_key, "embed")

    @contextmanager
    def get_lock(self, name: str) -> Iterator[None]:
        with self._redis.get_lock(f"lock:{name}"):
            yield

    def _get_embedding_key(self, name: str, mhash: MHash) -> str:
        return f"{self._redis.get_prefix()}:map:{name}:{mhash.to_parseable()}"

    def _get_staging_key(self, name: str) -> str:
        return f"{self._redis.get_prefix()}:staging:{name}"

    def _get_order_key(self, name: str) -> str:
        return f"{self._redis.get_prefix()}:order:{name}"

    def _serialize(self, embed: torch.Tensor) -> bytes:
        return pickle.dumps(embed, protocol=pickle.HIGHEST_PROTOCOL)

    def _deserialize(self, content: bytes) -> torch.Tensor:
        return pickle.loads(content)

    def set_map_embedding(
            self, name: str, mhash: MHash, embed: torch.Tensor) -> None:
        key = self._get_embedding_key(name, mhash)
        with self._redis.get_connection(depth=1) as conn:
            conn.set(key, self._serialize(embed))

    def get_map_embedding(
            self, name: str, mhash: MHash) -> torch.Tensor | None:
        key = self._get_embedding_key(name, mhash)
        with self._redis.get_connection(depth=1) as conn:
            res = conn.get(key)
        if res is None:
            return None
        return self._deserialize(res)

    def get_entry_by_index(self, name: str, index: int) -> MHash:
        key = self._get_order_key(name)
        return self._get_index(key, index)

    def add_embedding(self, name: str, embed: torch.Tensor) -> int:
        key = self._get_order_key(name)
        return self._add_embedding(key, embed)

    def embedding_count(self, name: str) -> int:
        key = self._get_order_key(name)
        return self._embeddings_size(key)

    def embeddings(self, name: str) -> Iterable[tuple[int, torch.Tensor]]:
        key = self._get_order_key(name)
        return self._get_embeddigs(key)

    def clear_embeddings(self, name: str) -> None:
        key = self._get_order_key(name)
        self._clear_embeddings(key)

    def add_staging_embedding(
            self, name: str, embed: torch.Tensor) -> None:
        key = self._get_staging_key(name)
        return self._add_embedding(key, embed)

    def staging_embeddings(
            self, name: str) -> Iterable[tuple[int, torch.Tensor]]:
        key = self._get_staging_key(name)
        return self._get_embeddigs(key)

    def get_staging_entry_by_index(self, name: str, index: int) -> MHash:
        key = self._get_staging_key(name)
        return self._get_index(key, index)

    def staging_count(self, name: str) -> int:
        key = self._get_staging_key(name)
        return self._embeddings_size(key)

    def clear_staging(self, name: str) -> None:
        key = self._get_staging_key(name)
        self._clear_embeddings(key)

    def _add_embedding(self, key: str, embed: torch.Tensor) -> int:
        with self._redis.get_connection(depth=1) as conn:
            res = int(conn.rpush(key, self._serialize(embed)))
            return res - 1

    def _get_index(self, key: str, index: int) -> torch.Tensor:
        with self._redis.get_connection(depth=2) as conn:
            res = conn.lindex(key, index)
            if res is None:
                raise KeyError(f"index not in list: {key} {index}")
            return self._deserialize(res)

    def _get_embeddigs(self, key: str) -> Iterable[tuple[int, torch.Tensor]]:
        offset = 0
        batch_size = 100
        with self._redis.get_connection(depth=2) as conn:
            while True:
                res = conn.lrange(key, offset, offset + batch_size)
                if not res:
                    break
                yield from (
                    (offset + ix, self._deserialize(elem))
                    for ix, elem in enumerate(res)
                )
                offset += batch_size

    def _embeddings_size(self, key: str) -> int:
        with self._redis.get_connection(depth=2) as conn:
            return int(conn.llen(key))

    def _clear_embeddings(self, key: str) -> None:
        with self._redis.get_connection(depth=2) as conn:
            conn.delete(key)
