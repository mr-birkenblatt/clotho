from typing import Literal

import torch

from system.msgs.message import Message


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
