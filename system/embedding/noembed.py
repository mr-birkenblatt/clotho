from typing import Iterable

import torch

from model.embedding import ProviderRole
from system.embedding.store import EmbeddingStore
from system.msgs.message import MHash


class NoEmbedding(EmbeddingStore):
    def do_add_embedding(
            self,
            role: ProviderRole,
            mhash: MHash,
            embed: torch.Tensor,
            *,
            no_index: bool) -> None:
        pass

    def do_get_embedding(
            self,
            role: ProviderRole,
            mhash: MHash) -> torch.Tensor | None:
        return torch.Tensor([0])

    def do_get_closest(
            self,
            role: ProviderRole,
            embed: torch.Tensor,
            count: int,
            *,
            precise: bool) -> Iterable[MHash]:
        yield from []

    def get_all_embeddings(
            self,
            role: ProviderRole,
            *,
            progress_bar: bool) -> Iterable[tuple[MHash, torch.Tensor]]:
        yield from []

    def get_embedding_count(self, role: ProviderRole) -> int:
        return 0

    def self_test(self, role: ProviderRole, count: int | None) -> None:
        raise ValueError("no computation has happened; nothing to test")
