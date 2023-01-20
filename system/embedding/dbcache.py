import collections
import threading
from typing import Iterable

import numpy as np
import sqlalchemy as sa
import torch
from sqlalchemy.dialects.postgresql import insert as pg_insert

from db.base import Base, NamespaceTable
from db.db import DBConnector
from misc.util import safe_ravel
from model.embedding import EmbeddingProvider, ProviderEnum, ProviderRole
from system.embedding.index_lookup import EmbeddingCache
from system.embedding.store import EmbeddingStore
from system.msgs.message import MHash
from system.namespace.namespace import Namespace


MODULE_VERSION = 1


class EmbedTable(Base):  # pylint: disable=too-few-public-methods
    __tablename__ = "embed"

    namespace_id = sa.Column(
        sa.Integer,
        sa.ForeignKey(
            NamespaceTable.id, onupdate="CASCADE", ondelete="CASCADE"),
        primary_key=True)
    role = sa.Column(
        sa.Enum(ProviderEnum),
        primary_key=True,
        nullable=False)
    mhash = sa.Column(
        sa.String(MHash.parse_size()),
        primary_key=True,
        nullable=False)
    main_order = sa.Column(
        sa.Integer,
        autoincrement=True,
        nullable=False,
        unique=True)
    embedding = sa.Column(
        sa.ARRAY(sa.Float),
        nullable=False)

    idx_main_order = sa.Index("namespace_id", "role", "main_order")


class DBEmbeddingCache(EmbeddingCache):
    def __init__(
            self,
            namespace: Namespace,
            db: DBConnector) -> None:
        super().__init__()
        self._db = db
        self._namespace = namespace
        self._nid: int | None = None
        self._locks: collections.defaultdict[ProviderRole, threading.RLock] = \
            collections.defaultdict(threading.RLock)

    def _get_nid(self) -> int:
        nid = self._nid
        if nid is None:
            nid = self._db.get_namespace_id(self._namespace, create=True)
            self._nid = nid
        return nid

    def _get_ns_name(self) -> str:
        return self._namespace.get_name()

    @staticmethod
    def _to_tensor(arr: list[float]) -> torch.Tensor:
        return torch.DoubleTensor(list(arr))

    @staticmethod
    def _from_tensor(x: torch.Tensor) -> list[float]:
        return safe_ravel(x.double().detach()).numpy().astype(np.float64)

    @staticmethod
    def cache_name() -> str:
        return "db"

    def is_cache_init(self) -> bool:
        return self._db.is_module_init(
            EmbeddingStore, MODULE_VERSION, "dbcache")

    def initialize_cache(self) -> None:
        self._db.create_module_tables(
            EmbeddingStore, MODULE_VERSION, [EmbedTable], "dbcache")

    def set_map_embedding(
            self,
            provider: EmbeddingProvider,
            mhash: MHash,
            embed: torch.Tensor) -> None:
        with self._db.get_connection() as conn:
            values = {
                "namespace_id": self._get_nid(),
                "role": provider.get_enum(),
                "mhash": mhash.to_parseable(),
                "embedding": self._from_tensor(embed),
            }
            stmt = pg_insert(EmbedTable).values(values)
            stmt = stmt.on_conflict_do_nothing()
            conn.execute(stmt)

    def get_map_embedding(
            self,
            provider: EmbeddingProvider,
            mhash: MHash) -> torch.Tensor | None:
        with self._db.get_connection() as conn:
            # FIXME investigate type error
            ecol: sa.Column = EmbedTable.embedding  # type: ignore
            stmt = sa.select([ecol]).where(sa.and_(
                EmbedTable.namespace_id == self._get_nid(),
                EmbedTable.role == provider.get_enum(),
                EmbedTable.mhash == mhash.to_parseable(),
            ))
            res = conn.execute(stmt).scalar()
        return None if res is None else self._to_tensor(res)

    def get_entry_by_index(
            self, provider: EmbeddingProvider, index: int) -> MHash:
        with self._db.get_connection() as conn:
            return self._entry_by_index(conn, provider, index)

    def embedding_count(self, provider: EmbeddingProvider) -> int:
        with self._db.get_connection() as conn:
            return self._embedding_count(conn, provider)

    def embeddings(
            self,
            provider: EmbeddingProvider,
            *,
            start_ix: int,
            limit: int | None,
            ) -> Iterable[tuple[int, MHash, torch.Tensor]]:
        with self._db.get_connection() as conn:
            yield from self._iter_column(
                conn,
                provider,
                start_ix=start_ix,
                limit=limit)

    def _embedding_count(
            self,
            conn: sa.engine.Connection,
            provider: EmbeddingProvider) -> int:
        cstmt = sa.select([sa.func.count()]).select_from(EmbedTable).where(
            sa.and_(
                EmbedTable.namespace_id == self._get_nid(),
                EmbedTable.role == provider.get_enum(),
            ))
        count: int | None = conn.execute(cstmt).scalar()
        return 0 if count is None else count

    def _iter_column(
            self,
            conn: sa.engine.Connection,
            provider: EmbeddingProvider,
            *,
            start_ix: int,
            limit: int | None,
            ) -> Iterable[tuple[int, MHash, torch.Tensor]]:
        # FIXME investigate type error
        ecol: sa.Column = EmbedTable.embedding  # type: ignore
        stmt = sa.select([EmbedTable.mhash, ecol]).where(sa.and_(
            EmbedTable.namespace_id == self._get_nid(),
            EmbedTable.role == provider.get_enum(),
        ))
        stmt = stmt.order_by(EmbedTable.main_order.asc())
        stmt = stmt.offset(start_ix)
        if limit is not None:
            stmt = stmt.limit(limit)
        for ix, row in enumerate(conn.execute(stmt)):
            cur_mhash = MHash.parse(row.mhash)
            cur_embed = self._to_tensor(row.embed)
            cur_ix = start_ix + ix
            yield (cur_ix, cur_mhash, cur_embed)

    def _entry_by_index(
            self,
            conn: sa.engine.Connection,
            provider: EmbeddingProvider,
            index: int) -> MHash:
        stmt = sa.select([EmbedTable.mhash]).where(sa.and_(
            EmbedTable.namespace_id == self._get_nid(),
            EmbedTable.role == provider.get_enum(),
        ))
        stmt = stmt.order_by(EmbedTable.main_order.asc())
        stmt = stmt.offset(index)
        stmt = stmt.limit(1)
        res = conn.execute(stmt).scalar()
        if res is None:
            raise IndexError(
                f"{index} not in entry db "
                f"({self._get_ns_name()};{provider.get_role()})")
        return MHash.parse(res)
