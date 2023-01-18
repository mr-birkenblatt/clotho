import collections
import contextlib
import threading
from typing import Callable, Iterable, Iterator

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
    main_order = sa.Column(sa.Integer, nullable=True)
    staging_order = sa.Column(sa.Integer, nullable=True)
    embedding = sa.Column(
        sa.ARRAY(sa.Float),
        nullable=False)

    idx_main_order = sa.Index("namespace_id", "role", "main_order")
    idx_staging_order = sa.Index("namespace_id", "role", "staging_order")


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

    @contextlib.contextmanager
    def get_lock(self, provider: EmbeddingProvider) -> Iterator[None]:
        with self._locks[provider.get_role()]:
            yield

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
                "main_order": None,
                "staging_order": None,
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
            return self._entry_by_index(
                conn, provider, EmbedTable.main_order, "", index)

    def add_embedding(
            self, provider: EmbeddingProvider, mhash: MHash) -> None:

        def get_next_ix(conn: sa.engine.Connection) -> int:
            return self._embedding_count(conn, provider, EmbedTable.main_order)

        self._add_embedding(
            provider,
            mhash,
            EmbedTable.main_order,
            col_name="main_order",
            ctx="",
            get_next_ix=get_next_ix)

    def embedding_count(self, provider: EmbeddingProvider) -> int:
        with self._db.get_connection() as conn:
            return self._embedding_count(conn, provider, EmbedTable.main_order)

    def embeddings(
            self,
            provider: EmbeddingProvider,
            *,
            start_ix: int,
            ) -> Iterable[tuple[int, MHash, torch.Tensor]]:
        with self._db.get_connection() as conn:
            yield from self._iter_column(
                conn,
                provider,
                EmbedTable.main_order,
                start_ix=start_ix,
                limit=None,
                return_real_index=False)

    def clear_embeddings(self, provider: EmbeddingProvider) -> None:
        self._clear_column(provider, EmbedTable.main_order, "main_order")

    def add_staging_embedding(
            self, provider: EmbeddingProvider, mhash: MHash) -> None:

        def get_next_ix(conn: sa.engine.Connection) -> int:
            high_ix = self._first_column_index(
                conn, provider, EmbedTable.staging_order, reverse=True)
            return 0 if high_ix is None else high_ix + 1

        self._add_embedding(
            provider,
            mhash,
            EmbedTable.staging_order,
            col_name="staging_order",
            ctx="staging ",
            get_next_ix=get_next_ix)

    def staging_embeddings(
            self,
            provider: EmbeddingProvider,
            *,
            remove: bool,
            ) -> Iterable[tuple[int, MHash, torch.Tensor]]:
        with self._db.get_connection() as conn:
            if not remove:
                yield from self._iter_column(
                    conn,
                    provider,
                    EmbedTable.staging_order,
                    start_ix=0,
                    limit=None,
                    return_real_index=False)
                return
            while True:
                with conn.begin() as trans:
                    row = self._get_lowest_staging_entry(conn, provider)
                    if row is None:
                        trans.commit()
                        break
                    low_ix, mhash, embed = row
                    yield (0, mhash, embed)
                    stmt = sa.update(EmbedTable).where(sa.and_(
                        EmbedTable.namespace_id == self._get_nid(),
                        EmbedTable.role == provider.get_enum(),
                        EmbedTable.staging_order == low_ix,
                    )).values({
                        EmbedTable.staging_order: None,
                    })
                    conn.execute(stmt)
                    trans.commit()

    def _get_lowest_staging_entry(
            self,
            conn: sa.engine.Connection,
            provider: EmbeddingProvider,
            ) -> tuple[int, MHash, torch.Tensor] | None:
        res = None
        for row in self._iter_column(
                conn,
                provider,
                EmbedTable.staging_order,
                start_ix=0,
                limit=1,
                return_real_index=True):
            res = row
        return res

    def get_staging_entry_by_index(
            self, provider: EmbeddingProvider, index: int) -> MHash:
        with self._db.get_connection() as conn:
            return self._entry_by_index(
                conn, provider, EmbedTable.staging_order, "staging ", index)

    def staging_count(self, provider: EmbeddingProvider) -> int:
        with self._db.get_connection() as conn:
            return self._embedding_count(
                conn, provider, EmbedTable.staging_order)

    def clear_staging(self, provider: EmbeddingProvider) -> None:
        self._clear_column(provider, EmbedTable.staging_order, "staging_order")

    def _clear_column(
            self,
            provider: EmbeddingProvider,
            col: sa.Column,
            col_name: str) -> None:
        with self._db.get_connection() as conn:
            stmt = sa.update(EmbedTable).where(sa.and_(
                EmbedTable.namespace_id == self._get_nid(),
                EmbedTable.role == provider.get_enum(),
                col.is_not(None),
            )).values({
                col_name: None,
            })
            conn.execute(stmt)

    def _embedding_count(
            self,
            conn: sa.engine.Connection,
            provider: EmbeddingProvider,
            order_col: sa.Column) -> int:
        cstmt = sa.select([sa.func.count()]).select_from(EmbedTable).where(
            sa.and_(
                EmbedTable.namespace_id == self._get_nid(),
                EmbedTable.role == provider.get_enum(),
                order_col.is_not(None),
            ))
        count: int | None = conn.execute(cstmt).scalar()
        return 0 if count is None else count

    def _iter_column(
            self,
            conn: sa.engine.Connection,
            provider: EmbeddingProvider,
            col: sa.Column,
            *,
            start_ix: int,
            limit: int | None,
            return_real_index: bool
            ) -> Iterable[tuple[int, MHash, torch.Tensor]]:
        # FIXME investigate type error
        ecol: sa.Column = EmbedTable.embedding  # type: ignore
        stmt = sa.select([col, EmbedTable.mhash, ecol]).where(sa.and_(
            EmbedTable.namespace_id == self._get_nid(),
            EmbedTable.role == provider.get_enum(),
            col.is_not(None),
        ))
        stmt = stmt.order_by(col.asc())
        stmt = stmt.offset(start_ix)
        if limit is not None:
            stmt = stmt.limit(limit)
        for ix, row in enumerate(conn.execute(stmt)):
            cur_mhash = MHash.parse(row.mhash)
            cur_embed = self._to_tensor(row.embed)
            cur_ix = row[0] if return_real_index else start_ix + ix
            yield (cur_ix, cur_mhash, cur_embed)

    def _first_column_index(
            self,
            conn: sa.engine.Connection,
            provider: EmbeddingProvider,
            col: sa.Column,
            *,
            reverse: bool
            ) -> int | None:
        stmt = sa.select([col]).where(sa.and_(
            EmbedTable.namespace_id == self._get_nid(),
            EmbedTable.role == provider.get_enum(),
            col.is_not(None),
        ))
        stmt = stmt.order_by(col.desc() if reverse else col.asc())
        stmt = stmt.limit(1)
        return conn.execute(stmt).scalar()

    def _entry_by_index(
            self,
            conn: sa.engine.Connection,
            provider: EmbeddingProvider,
            col: sa.Column,
            ctx: str,
            index: int) -> MHash:
        stmt = sa.select([EmbedTable.mhash]).where(sa.and_(
            EmbedTable.namespace_id == self._get_nid(),
            EmbedTable.role == provider.get_enum(),
            col.is_not(None),
        ))
        stmt = stmt.order_by(col.asc())
        stmt = stmt.offset(index).limit(1)
        res = conn.execute(stmt).scalar()
        if res is None:
            raise IndexError(
                f"{index} not in {ctx}entry db "
                f"({self._get_ns_name()};{provider.get_role()})")
        return MHash.parse(res)

    def _add_embedding(
            self,
            provider: EmbeddingProvider,
            mhash: MHash,
            col: sa.Column,
            *,
            col_name: str,
            ctx: str,
            get_next_ix: Callable[[sa.engine.Connection], int]) -> None:
        with self._db.get_connection() as conn:
            with conn.begin() as trans:
                high_ix = get_next_ix(conn)
                stmt = sa.update(EmbedTable).where(sa.and_(
                    EmbedTable.namespace_id == self._get_nid(),
                    EmbedTable.role == provider.get_enum(),
                    EmbedTable.mhash == mhash.to_parseable(),
                    col.is_(None),
                )).values({
                    col_name: high_ix,
                })
                res = conn.execute(stmt)
                if res.rowcount != 1:
                    raise ValueError(
                        f"{ctx}item {mhash} already "
                        f"added or not found ({res.rowcount}): "
                        f"({self._get_ns_name()};{provider.get_role()})")
                trans.commit()
