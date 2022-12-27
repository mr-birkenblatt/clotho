from contextlib import contextmanager
from typing import Iterable, Iterator

import torch

from system.embedding.store import EmbeddingStore
from system.msgs.message import MHash


class NoEmbedding(EmbeddingStore):
    def do_add_embedding(
            self,
            name: str,
            mhash: MHash,
            embed: torch.Tensor) -> None:
        pass

    def do_get_embedding(
            self,
            name: str,
            mhash: MHash) -> torch.Tensor | None:
        return torch.Tensor([0])

    @contextmanager
    def bulk_add(self, name: str) -> Iterator[None]:
        yield

    def do_get_closest(
            self, name: str, embed: torch.Tensor) -> Iterable[MHash]:
        yield from []
