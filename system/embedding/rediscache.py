import gzip
import io
from typing import Iterable

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
        self._keys: list[str] = []

    @staticmethod
    def cache_name() -> str:
        return "redis"

    def get_embedding_id_for(self, provider: EmbeddingProvider) -> int:
        name = provider.get_redis_name()
        try:
            return self._keys.index(name)
        except ValueError:
            pass
        embedding_id = len(self._keys)
        self._keys.append(name)
        return embedding_id

    def _get_embedding_key(
            self,
            embedding_id: int,
            mhash: MHash) -> str:
        name = self._keys[embedding_id]
        return f"{self._redis.get_prefix()}:map:{name}:{mhash.to_parseable()}"

    def _get_order_key(self, embedding_id: int) -> str:
        name = self._keys[embedding_id]
        return f"{self._redis.get_prefix()}:order:{name}"

    def _serialize(self, embed: torch.Tensor) -> bytes:
        bout = io.BytesIO()
        with gzip.GzipFile(fileobj=bout, mode="w") as fout:
            np.save(fout, embed.double().detach().numpy().astype(np.float64))
        return bout.getvalue()

    def _deserialize(self, content: bytes) -> torch.Tensor:
        binp = io.BytesIO(content)
        with gzip.GzipFile(fileobj=binp, mode="r") as finp:
            return torch.DoubleTensor(np.load(finp))

    def set_map_embedding(
            self,
            embedding_id: int,
            mhash: MHash,
            embed: torch.Tensor) -> None:
        key = self._get_embedding_key(embedding_id, mhash)
        with self._redis.get_connection(depth=0) as conn:
            with conn.pipeline() as pipe:
                pipe.exists(key)
                pipe.set(key, self._serialize(embed))
                has, _ = pipe.execute()
            if not int(has):
                conn.rpush(key, mhash.to_parseable().encode("utf-8"))

    def get_map_embedding(
            self, embedding_id: int, mhash: MHash) -> torch.Tensor | None:
        key = self._get_embedding_key(embedding_id, mhash)
        with self._redis.get_connection(depth=0) as conn:
            res = conn.get(key)
        if res is None:
            return None
        return self._deserialize(res)

    def get_entry_by_index(self, embedding_id: int, *, index: int) -> MHash:
        key = self._get_order_key(embedding_id)
        return self._get_index(key, index)

    def embedding_count(self, embedding_id: int) -> int:
        key = self._get_order_key(embedding_id)
        return self._embeddings_size(key)

    def embeddings(
            self,
            embedding_id: int,
            *,
            start_ix: int,
            limit: int | None,
            ) -> Iterable[tuple[int, MHash, torch.Tensor]]:
        key = self._get_order_key(embedding_id)
        return self._get_embeddigs(
            key, embedding_id, start_ix=start_ix, limit=limit)

    def _get_index(self, key: str, index: int) -> MHash:
        with self._redis.get_connection(depth=1) as conn:
            res = conn.lindex(key, index)
            if res is None:
                raise KeyError(f"index not in list: {key} {index}")
            return MHash.parse(res.decode("utf-8"))

    def _get_embeddigs(
            self,
            key: str,
            embedding_id: int,
            *,
            start_ix: int,
            limit: int | None,
            ) -> Iterable[tuple[int, MHash, torch.Tensor]]:
        offset = start_ix
        batch_size = 100

        def as_mhash(elem: bytes) -> MHash:
            return MHash.parse(elem.decode("utf-8"))

        def as_tensor(mhash: MHash) -> torch.Tensor:
            tres = self.get_map_embedding(embedding_id, mhash)
            if tres is None:
                raise KeyError(f"missing key: {mhash}")
            return tres

        def as_tuple(
                offset: int,
                ix: int,
                elem: bytes) -> tuple[int, MHash, torch.Tensor]:
            mhash = as_mhash(elem)
            return (offset + ix, mhash, as_tensor(mhash))

        remain = limit
        with self._redis.get_connection(depth=1) as conn:
            while True:
                if remain is not None and remain <= 0:
                    break
                res = conn.lrange(key, offset, offset + batch_size)
                if not res:
                    break
                tmp = [
                    as_tuple(offset, ix, elem)
                    for ix, elem in enumerate(res)
                ]
                if remain is not None:
                    tmp = tmp[:remain]
                    remain -= len(tmp)
                yield from tmp
                offset += len(tmp)

    def _embeddings_size(self, key: str) -> int:
        with self._redis.get_connection(depth=1) as conn:
            return int(conn.llen(key))
