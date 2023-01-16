import os
from typing import get_args, Literal, TypedDict

import torch

from system.msgs.message import Message
from system.namespace.namespace import Namespace


ProviderRole = Literal["parent", "child"]
PROVIDER_ROLES: list[ProviderRole] = list(get_args(ProviderRole))


class EmbeddingProvider:
    def __init__(self, method: str, role: ProviderRole) -> None:
        self._redis_name = f"{method}:{role}"
        self._file_name = f"{method}.{role}"

    def get_redis_name(self) -> str:
        return self._redis_name

    def get_file_name(self) -> str:
        return self._file_name

    def get_embedding(self, msg: Message) -> torch.Tensor:
        raise NotImplementedError()

    @staticmethod
    def num_dimensions() -> int:
        raise NotImplementedError()


class NoEmbeddingProvider(EmbeddingProvider):
    def get_embedding(self, msg: Message) -> torch.Tensor:
        return torch.Tensor([0])

    @staticmethod
    def num_dimensions() -> int:
        return 1


EmbeddingProviderMap = TypedDict('EmbeddingProviderMap', {
    "parent": EmbeddingProvider,
    "child": EmbeddingProvider,
})


PROVIDER_CACHE: dict[Namespace, EmbeddingProviderMap] = {}


def get_embed_providers(namespace: Namespace) -> EmbeddingProviderMap:
    res = PROVIDER_CACHE.get(namespace)
    if res is None:
        res = create_embed_providers(namespace)
        PROVIDER_CACHE[namespace] = res
    return res


TransformerEmbeddingModule = TypedDict('TransformerEmbeddingModule', {
    "name": Literal["transformer"],
    "fname": str,
    "version": int,
    "is_harness": bool,
})
NoEmbeddingModule = TypedDict('NoEmbeddingModule', {
    "name": Literal["none"],
})
EmbeddingProviderModule = TransformerEmbeddingModule | NoEmbeddingModule


def create_embed_providers(namespace: Namespace) -> EmbeddingProviderMap:
    pobj = namespace.get_embedding_providers()
    if pobj["name"] == "transformer":
        from model.transformer_embed import load_providers

        return load_providers(
            os.path.join(namespace.get_root(), "model"),
            pobj["fname"],
            pobj["version"],
            pobj["is_harness"])
    if pobj["name"] == "none":
        return {
            "parent": NoEmbeddingProvider("none", "parent"),
            "child": NoEmbeddingProvider("none", "child"),
        }
    raise ValueError(f"unknown embed provider: {pobj}")
