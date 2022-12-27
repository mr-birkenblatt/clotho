from typing import Literal, TypedDict

import torch

from system.msgs.message import Message
from system.namespace.namespace import Namespace


class EmbeddingProvider:
    def __init__(self, method: str, role: Literal["parent", "child"]) -> None:
        self._name = f"{method}:{role}"

    def get_name(self) -> str:
        return self._name

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


PROVIDER_CACHE: dict[Namespace, list[EmbeddingProvider]] = {}


def get_embed_providers(namespace: Namespace) -> list[EmbeddingProvider]:
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


def create_embed_providers(namespace: Namespace) -> list[EmbeddingProvider]:
    pobj = namespace.get_embedding_providers()
    if pobj["name"] == "transformer":
        from model.transformer_embed import load_providers

        return load_providers(
            "model", pobj["fname"], pobj["version"], pobj["is_harness"])
    if pobj["name"] == "none":
        return [
            NoEmbeddingProvider("none", "parent"),
            NoEmbeddingProvider("none", "child"),
        ]
    raise ValueError(f"unknown embed provider: {pobj}")
