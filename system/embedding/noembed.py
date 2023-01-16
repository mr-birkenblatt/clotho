from contextlib import contextmanager
from typing import Iterable, Iterator

import torch

from model.embedding import ProviderRole
from system.embedding.store import EmbeddingStore
from system.msgs.message import MHash


class NoEmbedding(EmbeddingStore):
    def is_module_init(self) -> bool:
        return True

    def do_add_embedding(
            self,
            role: ProviderRole,
            mhash: MHash,
            embed: torch.Tensor) -> None:
        pass

    def do_get_embedding(
            self,
            role: ProviderRole,
            mhash: MHash) -> torch.Tensor | None:
        return torch.Tensor([0])

    @contextmanager
    def bulk_add(self, role: ProviderRole) -> Iterator[None]:
        yield

    def do_get_closest(
            self,
            role: ProviderRole,
            embed: torch.Tensor,
            count: int,
            *,
            precise: bool) -> Iterable[MHash]:
        yield from []

    def self_test(self, role: ProviderRole, count: int | None) -> None:
        raise ValueError("no computation has happened; nothing to test")
