import collections
import contextlib
import os
import shutil
import threading
from typing import IO, Iterable, Iterator

import numpy as np
import sqlalchemy as sa
import torch

from db.base import Base, MHashTable, NamespaceTable
from db.db import DBConnector
from misc.io import ensure_folder, open_read
from misc.util import file_hash_size, get_file_hash, safe_ravel
from model.embedding import EmbeddingProvider, ProviderEnum, ProviderRole
from system.embedding.index_lookup import EmbeddingCache
from system.embedding.store import EmbeddingStore
from system.msgs.message import MHash
from system.namespace.namespace import Namespace


MODULE_VERSION = 2


class ModelsTable(Base):  # pylint: disable=too-few-public-methods
    __tablename__ = "models"

    id = sa.Column(
        sa.Integer,
        primary_key=True,
        autoincrement=True,
        nullable=False,
        unique=True)
    model_hash = sa.Column(
        sa.String(file_hash_size()),
        nullable=False,
        unique=True)
    name = sa.Column(sa.String)
    version = sa.Column(sa.Integer)
    is_harness = sa.Column(sa.Boolean)

    idx_model_hash = sa.Index("model_hash")


EMBED_CONFIG_ID: sa.Sequence = sa.Sequence(
    "embed_config_id_seq", start=1, increment=1)


class EmbedConfigTable(Base):  # pylint: disable=too-few-public-methods
    __tablename__ = "embedconfig"

    namespace_id = sa.Column(
        sa.Integer,
        sa.ForeignKey(
            NamespaceTable.id, onupdate="CASCADE", ondelete="CASCADE"),
        primary_key=True)
    role = sa.Column(
        sa.Enum(ProviderEnum),
        primary_key=True,
        nullable=False)
    model_id = sa.Column(
        sa.Integer,
        sa.ForeignKey(
            ModelsTable.id, onupdate="CASCADE", ondelete="CASCADE"),
        primary_key=True)
    config_id = sa.Column(
        sa.Integer,
        EMBED_CONFIG_ID,
        nullable=False,
        unique=True,
        server_default=EMBED_CONFIG_ID.next_value())

    idx_config_id = sa.Index("config_id")

    # namespace = sa.orm.relationship(
    #     NamespaceTable,
    #     back_populates="embedconfig",
    #     uselist=False,
    #     primaryjoin=namespace_id == NamespaceTable.id,
    #     foreign_keys=namespace_id)
    # models = sa.orm.relationship(
    #     ModelsTable,
    #     back_populates="embedconfig",
    #     uselist=False,
    #     primaryjoin=model_id == ModelsTable.id,
    #     foreign_keys=model_id)


# NamespaceTable.embedconfig = sa.orm.relationship(
#     EmbedConfigTable, back_populates="namespace", uselist=False)
# ModelsTable.embedconfig = sa.orm.relationship(
#     EmbedConfigTable, back_populates="models", uselist=False)


MAIN_ORDER_SEQ: sa.Sequence = sa.Sequence(
    "main_order_seq", start=1, increment=1)


class EmbedTable(Base):  # pylint: disable=too-few-public-methods
    __tablename__ = "embed"

    config_id = sa.Column(
        sa.Integer,
        sa.ForeignKey(
            EmbedConfigTable.config_id,
            onupdate="CASCADE",
            ondelete="CASCADE"),
        primary_key=True)
    mhash_id = sa.Column(
        sa.Integer,
        sa.ForeignKey(
            MHashTable.id, onupdate="CASCADE", ondelete="CASCADE"),
        primary_key=True)
    main_order = sa.Column(
        sa.Integer,
        MAIN_ORDER_SEQ,
        nullable=False,
        unique=True,
        server_default=MAIN_ORDER_SEQ.next_value())
    embedding = sa.Column(
        sa.ARRAY(sa.Float),
        nullable=False)

    idx_main_order = sa.Index("config_id", "main_order")

    # embedconfig = sa.orm.relationship(
    #     EmbedConfigTable,
    #     back_populates="embed",
    #     uselist=False,
    #     primaryjoin=config_id == EmbedConfigTable.config_id,
    #     foreign_keys=config_id)
    # mhashes = sa.orm.relationship(
    #     MHashTable,
    #     back_populates="embed",
    #     uselist=False,
    #     primaryjoin=mhash_id == MHashTable.id,
    #     foreign_keys=mhash_id)


# EmbedConfigTable.embed = sa.orm.relationship(
#     EmbedTable, back_populates="embedconfig", uselist=False)
# MHashTable.embed = sa.orm.relationship(
#     EmbedTable, back_populates="mhashes", uselist=False)


def is_cache_init(db: DBConnector) -> bool:
    return db.is_module_init(EmbeddingStore, MODULE_VERSION, "dbcache")


def initialize_cache(db: DBConnector, *, force: bool) -> None:
    db.create_module_tables(
        EmbeddingStore,
        MODULE_VERSION,
        [EmbedTable, EmbedConfigTable, ModelsTable],
        "dbcache",
        force=force)


def model_registry(model_hash: str) -> str:
    root = ensure_folder(Namespace.get_root_for("_models"))
    return os.path.join(root, f"{model_hash}.pkl")


def register_model(
        db: DBConnector,
        root: str,
        fname: str,
        version: int,
        is_harness: bool) -> str:
    if not is_cache_init(db):
        initialize_cache(db, force=False)
    with db.get_session() as session:
        model_file = os.path.join(root, fname)
        model_hash = get_file_hash(model_file)
        model_name = fname
        rix = model_name.rfind(".")
        if rix >= 0:
            model_name = model_name[:rix]
        session.add(ModelsTable(
            model_hash=model_hash,
            name=model_name,
            version=version,
            is_harness=is_harness))
        try:
            shutil.copyfile(model_file, model_registry(model_hash))
        except shutil.SameFileError:
            pass
    return model_hash


@contextlib.contextmanager
def read_db_model(
        db: DBConnector,
        model_hash: str) -> Iterator[tuple[IO[bytes], str, int, bool]]:
    with db.get_connection() as conn:
        stmt = sa.select(
                ModelsTable.name,
                ModelsTable.version,
                ModelsTable.is_harness,
            ).where(ModelsTable.model_hash == model_hash)
        row = conn.execute(stmt).one()
        model_name = row.name
        version = row.version
        is_harness = row.is_harness
    with open_read(model_registry(model_hash), text=False) as fout:
        yield fout, model_name, version, is_harness


class DBEmbeddingCache(EmbeddingCache):
    def __init__(
            self,
            namespace: Namespace,
            db: DBConnector) -> None:
        super().__init__()
        self._db = db
        self._namespace = namespace
        self._nid: int | None = None
        self._model_ids: dict[ProviderRole, int] = {}
        self._ctx_map: dict[ProviderRole, int] = {}
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
        return is_cache_init(self._db)

    def initialize_cache(self, *, force: bool) -> None:
        initialize_cache(self._db, force=force)

    def _get_model_id(
            self, session: sa.orm.Session, provider: EmbeddingProvider) -> int:
        role = provider.get_role()
        res = self._model_ids.get(role)
        if res is None:
            model_hash = provider.get_embedding_hash()
            stmt = sa.select(ModelsTable.id).where(
                ModelsTable.model_hash == model_hash)
            res = session.execute(stmt).scalar()
            if res is None:
                raise ValueError(f"no model found for '{model_hash}' ({role})")
            self._model_ids[role] = res
        return res

    def _get_embedding_id_for(self, provider: EmbeddingProvider) -> int:
        nid = self._get_nid()
        role = provider.get_enum()

        def get_embedding_id(
                session: sa.orm.Session, model_id: int) -> int | None:
            stmt = sa.select(EmbedConfigTable.config_id).where(sa.and_(
                EmbedConfigTable.namespace_id == nid,
                EmbedConfigTable.role == role,
                EmbedConfigTable.model_id == model_id))
            return session.execute(stmt).scalar()

        with self._db.get_session() as session:
            model_id = self._get_model_id(session, provider)
            res = get_embedding_id(session, model_id)
            if res is None:
                session.add(EmbedConfigTable(
                    namespace_id=nid, role=role, model_id=model_id))
                session.commit()
                res = get_embedding_id(session, model_id)
                if res is None:
                    raise ValueError(
                        "error while adding config: "
                        f"ns={self._get_ns_name()} "
                        f"role={role} "
                        f"model_hash={provider.get_embedding_hash()}")
        return res

    def get_embedding_id_for(self, provider: EmbeddingProvider) -> int:
        role = provider.get_role()
        res = self._ctx_map.get(role)
        if res is None:
            res = self._get_embedding_id_for(provider)
            self._ctx_map[role] = res
        return res

    def set_map_embedding(
            self,
            embedding_id: int,
            mhash: MHash,
            embed: torch.Tensor) -> None:
        with self._db.get_session() as session:
            values = {
                "config_id": embedding_id,
                "mhash_id": self._db.get_mhash_id(
                    session, mhash, likely_exists=True),
                "embedding": self._from_tensor(embed),
            }
            stmt = self._db.upsert(EmbedTable).values(values)
            stmt = stmt.on_conflict_do_nothing()
            session.execute(stmt)

    def get_map_embedding(
            self, embedding_id: int, mhash: MHash) -> torch.Tensor | None:
        with self._db.get_connection() as conn:
            stmt = sa.select(EmbedTable.embedding).where(sa.and_(
                EmbedTable.config_id == embedding_id,
                EmbedTable.mhash_id == MHashTable.id,
                MHashTable.mhash == mhash.to_parseable(),
            ))
            res = conn.execute(stmt).scalar()
        return None if res is None else self._to_tensor(res)

    def get_entry_by_index(self, embedding_id: int, *, index: int) -> MHash:
        with self._db.get_connection() as conn:
            return self._entry_by_index(conn, embedding_id, index=index)

    def embedding_count(self, embedding_id: int) -> int:
        with self._db.get_connection() as conn:
            return self._embedding_count(conn, embedding_id)

    def embeddings(
            self,
            embedding_id: int,
            *,
            start_ix: int,
            limit: int | None,
            ) -> Iterable[tuple[int, MHash, torch.Tensor]]:
        with self._db.get_connection() as conn:
            yield from self._iter_column(
                conn,
                embedding_id,
                start_ix=start_ix,
                limit=limit)

    def _embedding_count(
            self, conn: sa.engine.Connection, embedding_id: int) -> int:
        cstmt = sa.select(sa.func.count()).select_from(EmbedTable).where(
            EmbedTable.config_id == embedding_id)
        count: int | None = conn.execute(cstmt).scalar()
        return 0 if count is None else count

    def _iter_column(
            self,
            conn: sa.engine.Connection,
            embedding_id: int,
            *,
            start_ix: int,
            limit: int | None,
            ) -> Iterable[tuple[int, MHash, torch.Tensor]]:
        stmt = sa.select(MHashTable.mhash, EmbedTable.embedding).where(sa.and_(
            EmbedTable.config_id == embedding_id,
            EmbedTable.mhash_id == MHashTable.id))
        stmt = stmt.order_by(EmbedTable.main_order.asc())
        stmt = stmt.offset(start_ix)
        if limit is not None:
            stmt = stmt.limit(limit)
        for ix, row in enumerate(conn.execute(stmt)):
            cur_mhash = MHash.parse(row.mhash)
            cur_embed = self._to_tensor(row.embedding)
            cur_ix = start_ix + ix
            yield (cur_ix, cur_mhash, cur_embed)

    def _entry_by_index(
            self,
            conn: sa.engine.Connection,
            embedding_id: int,
            *,
            index: int) -> MHash:
        stmt = sa.select(MHashTable.mhash).where(sa.and_(
            EmbedTable.config_id == embedding_id,
            EmbedTable.mhash_id == MHashTable.id))
        stmt = stmt.order_by(EmbedTable.main_order.asc())
        stmt = stmt.offset(index)
        stmt = stmt.limit(1)
        res = conn.execute(stmt).scalar()
        if res is None:
            raise IndexError(
                f"{index} not in entry db "
                f"(ns={self._get_ns_name()} config_id={embedding_id})")
        return MHash.parse(res)
