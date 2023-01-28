import time
from typing import Callable, Iterable

import sqlalchemy as sa

from db.base import Base, MHashTable, NamespaceTable
from db.db import DBConnector
from misc.lru import LRU
from misc.util import escape, unescape
from system.msgs.message import Message, MHash
from system.msgs.store import MessageStore, RandomGeneratingFunction
from system.namespace.namespace import Namespace


MODULE_VERSION = 2
RELOAD_TOPICS_FREQ = 60 * 60  # 1h
RELOAD_SIZE_FREQ = 60  # 1min


class MsgsTable(Base):  # pylint: disable=too-few-public-methods
    __tablename__ = "msgs"

    namespace_id = sa.Column(
        sa.Integer,
        sa.ForeignKey(
            NamespaceTable.id, onupdate="CASCADE", ondelete="CASCADE"),
        primary_key=True)
    mhash_id = sa.Column(
        sa.Integer,
        sa.ForeignKey(
            MHashTable.id, onupdate="CASCADE", ondelete="CASCADE"),
        primary_key=True)
    text = sa.Column(sa.Text, nullable=False)

    # namespace = sa.orm.relationship(
    #     NamespaceTable,
    #     back_populates="msgs",
    #     uselist=False,
    #     primaryjoin=namespace_id == NamespaceTable.id,
    #     foreign_keys=namespace_id)
    # mhashes = sa.orm.relationship(
    #     MHashTable,
    #     back_populates="msgs",
    #     uselist=False,
    #     primaryjoin=mhash_id == MHashTable.id,
    #     foreign_keys=mhash_id)


# NamespaceTable.msgs = sa.orm.relationship(
#     MsgsTable, back_populates="namespace", uselist=False)
# MHashTable.msgs = sa.orm.relationship(
#     MsgsTable, back_populates="mhashes", uselist=False)


class TopicsTable(Base):  # pylint: disable=too-few-public-methods
    __tablename__ = "topics"

    namespace_id = sa.Column(
        sa.Integer,
        sa.ForeignKey(
            NamespaceTable.id, onupdate="CASCADE", ondelete="CASCADE"),
        primary_key=True)
    id = sa.Column(
        sa.Integer,
        primary_key=True,
        autoincrement=True,
        nullable=False)
    mhash_id = sa.Column(
        sa.Integer,
        sa.ForeignKey(
            MHashTable.id, onupdate="CASCADE", ondelete="CASCADE"),
        primary_key=True)
    topic = sa.Column(sa.Text, nullable=False)

    # namespace = sa.orm.relationship(
    #     NamespaceTable,
    #     back_populates="topics",
    #     uselist=False,
    #     primaryjoin=namespace_id == NamespaceTable.id,
    #     foreign_keys=namespace_id)
    # mhashes = sa.orm.relationship(
    #     MHashTable,
    #     back_populates="topics",
    #     uselist=False,
    #     primaryjoin=mhash_id == MHashTable.id,
    #     foreign_keys=mhash_id)


# NamespaceTable.topics = sa.orm.relationship(
#     TopicsTable, back_populates="namespace", uselist=False)
# MHashTable.topics = sa.orm.relationship(
#     TopicsTable, back_populates="mhashes", uselist=False)


class DBStore(MessageStore):
    def __init__(
            self,
            namespace: Namespace,
            db: DBConnector,
            cache_size: int) -> None:
        super().__init__()
        self._db = db
        self._namespace = namespace
        self._nid: int | None = None
        self._cache: LRU[MHash, Message] = LRU(cache_size)
        self._topic_cache: list[Message] | None = None
        self._topic_update: float = 0.0
        self._size_cache: int | None = None
        self._min_cache: int | None = None
        self._max_cache: int | None = None
        self._size_update: float = 0.0

    def _get_nid(self) -> int:
        nid = self._nid
        if nid is None:
            nid = self._db.get_namespace_id(self._namespace, create=True)
            self._nid = nid
        return nid

    def _get_ns_name(self) -> str:
        return self._namespace.get_name()

    @staticmethod
    def _escape(text: str) -> str:
        return escape(text, {"\0": "0"})

    @staticmethod
    def _unescape(text: str) -> str:
        return unescape(text, {"0": "\0"})

    def is_module_init(self) -> bool:
        return self._db.is_module_init(self, MODULE_VERSION)

    def initialize_module(self, *, force: bool) -> None:
        self._db.create_module_tables(
            self, MODULE_VERSION, [MsgsTable, TopicsTable], force=force)

    def write_message(self, message: Message) -> MHash:
        mhash = message.get_hash()
        with self._db.get_session() as session:
            try:
                values = {
                    "namespace_id": self._get_nid(),
                    "mhash_id": self._db.get_mhash_id(
                        session, mhash, likely_exists=False),
                    "text": self._escape(message.get_text()),
                }
                stmt = self._db.upsert(MsgsTable).values(values)
                stmt = stmt.on_conflict_do_nothing()
                session.execute(stmt)
            except ValueError as e:
                raise ValueError(
                    "error while processing "
                    f"{message.get_hash()}: {repr(message.get_text())}") from e
        self._cache.set(mhash, message)
        return mhash

    def read_message(self, message_hash: MHash) -> Message:
        res = self._cache.get(message_hash)
        if res is not None:
            return res
        with self._db.get_connection() as conn:
            stmt = sa.select(MHashTable.mhash, MsgsTable.text).where(sa.and_(
                MsgsTable.namespace_id == self._get_nid(),
                MsgsTable.mhash_id == MHashTable.id,
                MHashTable.mhash == message_hash.to_parseable(),
            ))
            for row in conn.execute(stmt):
                cur_mhash = MHash.parse(row.mhash)
                cur_text = self._unescape(row.text)
                cur_msg = Message(msg=cur_text, msg_hash=cur_mhash)
                self._cache.set(cur_mhash, cur_msg)
                if cur_mhash == message_hash:
                    res = cur_msg
        if res is None:
            raise KeyError(f"{message_hash} not in db ({self._get_ns_name()})")
        return res

    def add_topic(self, topic: Message) -> MHash:
        mhash = topic.get_hash()
        with self._db.get_session() as session:
            values = {
                "namespace_id": self._get_nid(),
                "mhash_id": self._db.get_mhash_id(
                    session, mhash, likely_exists=False),
                "topic": self._escape(topic.get_text()),
            }
            stmt = self._db.upsert(TopicsTable).values(values)
            stmt = stmt.on_conflict_do_nothing()
            session.execute(stmt)
        self._topic_cache = None
        return mhash

    def _get_topics(self) -> list[Message]:
        res: dict[int, Message] = {}
        with self._db.get_connection() as conn:
            stmt = sa.select(
                TopicsTable.id, MHashTable.mhash, TopicsTable.topic,
                ).where(sa.and_(
                    TopicsTable.namespace_id == self._get_nid(),
                    TopicsTable.mhash_id == MHashTable.id))
            for row in conn.execute(stmt):
                cur_mhash = MHash.parse(row.mhash)
                cur_topic = self._unescape(row.topic)
                res[row.id] = Message(msg=cur_topic, msg_hash=cur_mhash)
        return [
            elem[1]
            for elem in sorted(res.items(), key=lambda elem: elem[0])
        ]

    def get_topics(
            self,
            offset: int,
            limit: int | None) -> list[Message]:
        cur_time = time.monotonic()
        if (self._topic_cache is None
                or cur_time >= self._topic_update + RELOAD_TOPICS_FREQ):
            self._topic_cache = self._get_topics()
            self._topic_update = cur_time
        if limit is None:
            return self._topic_cache[offset:]
        return self._topic_cache[offset:offset + limit]

    def get_topics_count(self) -> int:
        with self._db.get_connection() as conn:
            cstmt = sa.select(
                sa.func.count()).select_from(TopicsTable).where(
                TopicsTable.namespace_id == self._get_nid())
            count = conn.execute(cstmt).scalar()
        return 0 if count is None else count

    def do_get_random_messages(
            self,
            get_random: RandomGeneratingFunction,
            count: int) -> Iterable[MHash]:
        nid = self._get_nid()

        def slow_retrieve(
                conn: sa.engine.Connection,
                remain: int,
                total: int) -> Iterable[MHash]:
            offsets = [
                get_random(high=total, for_row=pos) + 1
                for pos in range(remain)
            ]
            row_id_col = sa.func.row_number().over().label("row_id")
            sub_stmt = sa.select(
                MsgsTable.mhash_id.label("mid"), row_id_col).where(
                    MsgsTable.namespace_id == nid)
            stmt = sa.select(
                MHashTable.mhash).select_from(MHashTable).join(
                    sub_stmt.subquery(), sa.Column("mid") == MHashTable.id)
            stmt = stmt.where(sa.Column("row_id").in_(offsets))
            for row in conn.execute(stmt):
                cur_mhash = MHash.parse(row.mhash)
                yield cur_mhash

        def fast_retrieve(
                conn: sa.engine.Connection,
                remain: int,
                mh_min: int,
                mh_max: int) -> Iterable[MHash]:
            offsets = [
                get_random(high=mh_max - mh_min + 1, for_row=pos) + mh_min
                for pos in range(remain)
            ]
            stmt = sa.select(
                MHashTable.mhash).select_from(MHashTable).join(
                    MsgsTable, MsgsTable.mhash_id == MHashTable.id)
            stmt = stmt.where(sa.and_(
                MsgsTable.mhash_id.in_(offsets),
                MsgsTable.namespace_id == nid))
            for row in conn.execute(stmt):
                cur_mhash = MHash.parse(row.mhash)
                yield cur_mhash

        with self._db.get_connection() as conn:
            total, mh_min, mh_max = self._get_stat(conn)
            if total is None or total == 0:
                yield from []
                return
            remain = count
            tries = 10  # 0
            while remain > 0 and tries > 0:
                if mh_min is None or mh_max is None:
                    break
                for mhash in fast_retrieve(conn, remain, mh_min, mh_max):
                    yield mhash
                    remain -= 1
                    if remain <= 0:
                        break
                tries -= 1
            if remain > 0:
                yield from slow_retrieve(conn, remain, total)

    def enumerate_messages(self, *, progress_bar: bool) -> Iterable[MHash]:

        def get_rows(
                conn: sa.engine.Connection,
                *,
                pbar: Callable[[], None] | None) -> Iterable[MHash]:
            stmt = sa.select(
                MHashTable.mhash, MsgsTable.text).where(sa.and_(
                    MsgsTable.namespace_id == self._get_nid(),
                    MsgsTable.mhash_id == MHashTable.id))
            for row in conn.execute(stmt):
                cur_mhash = MHash.parse(row.mhash)
                cur_text = self._unescape(row.text)
                cur_msg = Message(msg=cur_text, msg_hash=cur_mhash)
                self._cache.set(cur_mhash, cur_msg)
                yield cur_mhash
                if pbar is not None:
                    pbar()

        with self._db.get_connection() as conn:
            count = self._get_count(conn)
            if progress_bar is None or count is None:
                yield from get_rows(conn, pbar=None)
            else:
                # FIXME: add stubs
                from tqdm.auto import tqdm  # type: ignore

                with tqdm(total=count) as pbar:
                    yield from get_rows(conn, pbar=lambda: pbar.update(1))

    def get_message_count(self) -> int:
        with self._db.get_connection() as conn:
            count = self._get_count(conn)
        return 0 if count is None else count

    def _get_db_count(
            self,
            conn: sa.engine.Connection,
            ) -> tuple[int | None, int | None, int | None]:
        cstmt = sa.select(
                sa.func.min(MsgsTable.mhash_id).label("mh_min"),
                sa.func.max(MsgsTable.mhash_id).label("mh_max"),
                sa.func.count().label("count")).select_from(MsgsTable).where(
            MsgsTable.namespace_id == self._get_nid())
        row = conn.execute(cstmt).one_or_none()
        if row is None:
            return (None, None, None)
        return (row.count, row.mh_min, row.mh_max)

    def _ensure_count(self, conn: sa.engine.Connection) -> None:
        cur_time = time.monotonic()
        if (self._size_cache is None
                or cur_time >= self._size_update + RELOAD_SIZE_FREQ):
            res = self._get_db_count(conn)
            self._size_cache, self._min_cache, self._max_cache = res
            self._size_update = cur_time

    def _get_count(self, conn: sa.engine.Connection) -> int | None:
        self._ensure_count(conn)
        return self._size_cache

    def _get_stat(
            self,
            conn: sa.engine.Connection,
            ) -> tuple[int | None, int | None, int | None]:
        self._ensure_count(conn)
        return (self._size_cache, self._min_cache, self._max_cache)
