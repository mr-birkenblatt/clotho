import time
from typing import Callable, Iterable

import numpy as np
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import insert as pg_insert

from db.base import Base, NamespaceTable
from db.db import DBConnector
from misc.lru import LRU
from system.msgs.message import Message, MHash
from system.msgs.store import MessageStore
from system.namespace.namespace import Namespace


MODULE_VERSION = 1
RELOAD_TOPICS_FREQ = 60 * 60  # 1h


class MsgsTable(Base):  # pylint: disable=too-few-public-methods
    __tablename__ = "msgs"

    namespace_id = sa.Column(
        sa.Integer,
        sa.ForeignKey(NamespaceTable.id),
        primary_key=True)
    mhash = sa.Column(
        sa.String(MHash.parse_size()),
        primary_key=True,
        nullable=False)
    text = sa.Column(sa.Text, nullable=False)


class TopicsTable(Base):  # pylint: disable=too-few-public-methods
    __tablename__ = "topics"

    namespace_id = sa.Column(
        sa.Integer,
        sa.ForeignKey(NamespaceTable.id),
        primary_key=True)
    id = sa.Column(
        sa.Integer,
        primary_key=True,
        autoincrement=True,
        nullable=False)
    mhash = sa.Column(
        sa.String(MHash.parse_size()),
        primary_key=True,
        nullable=False)
    topic = sa.Column(sa.Text, nullable=False)


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
        return text.replace("\\", "\\\\").replace("\0", "\\0")

    @staticmethod
    def _unescape(text: str) -> str:
        return text.replace("\\0", "\0").replace("\\\\", "\\")

    def is_module_init(self) -> bool:
        return self._db.is_module_init(self, MODULE_VERSION)

    def initialize_module(self) -> None:
        self._db.create_module_tables(
            self, MODULE_VERSION, [MsgsTable, TopicsTable])

    def write_message(self, message: Message) -> MHash:
        mhash = message.get_hash()
        with self._db.get_connection() as conn:
            try:
                values = {
                    "namespace_id": self._get_nid(),
                    "mhash": mhash.to_parseable(),
                    "text": self._escape(message.get_text()),
                }
                stmt = pg_insert(MsgsTable).values(values)
                stmt = stmt.on_conflict_do_nothing()
                conn.execute(stmt)
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
            stmt = sa.select([MsgsTable.mhash, MsgsTable.text]).where(sa.and_([
                MsgsTable.namespace_id == self._get_nid(),
                MsgsTable.mhash == message_hash.to_parseable()
            ]))
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
        with self._db.get_connection() as conn:
            values = {
                "namespace_id": self._get_nid(),
                "mhash": mhash.to_parseable(),
                "topic": self._escape(topic.get_text()),
            }
            stmt = pg_insert(TopicsTable).values(values)
            stmt = stmt.on_conflict_do_nothing()
            conn.execute(stmt)
        self._topic_cache = None
        return mhash

    def _get_topics(self) -> list[Message]:
        res: dict[int, Message] = {}
        with self._db.get_connection() as conn:
            stmt = sa.select(
                [TopicsTable.id, TopicsTable.mhash, TopicsTable.topic],
                ).where(TopicsTable.namespace_id == self._get_nid())
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

    def do_get_random_messages(
            self, rng: np.random.Generator, count: int) -> Iterable[MHash]:
        raise RuntimeError("random messages are not supported in db yet")

    def enumerate_messages(self, *, progress_bar: bool) -> Iterable[MHash]:
        chunk_size = 1000

        def get_rows(
                conn: sa.engine.Connection,
                *,
                pbar: Callable[[], None] | None) -> Iterable[MHash]:
            offset = 0
            while True:
                stmt = sa.select(
                    [MsgsTable.mhash, MsgsTable.text]).where(
                    MsgsTable.namespace_id == self._get_nid(),
                    ).offset(offset).limit(chunk_size)
                had_data = False
                for row in conn.execute(stmt):
                    cur_mhash = MHash.parse(row.mhash)
                    cur_text = self._unescape(row.text)
                    cur_msg = Message(msg=cur_text, msg_hash=cur_mhash)
                    self._cache.set(cur_mhash, cur_msg)
                    yield cur_mhash
                    offset += 1
                    if pbar is not None:
                        pbar()
                    had_data = True
                if not had_data:
                    break

        with self._db.get_connection() as conn:
            cstmt = sa.select([sa.func.count()]).select_from(MsgsTable).where(
                MsgsTable.namespace_id == self._get_nid())
            count: int | None = conn.execute(cstmt).scalar()
            if progress_bar is None or count is None:
                yield from get_rows(conn, pbar=None)
            else:
                # FIXME: add stubs
                from tqdm.auto import tqdm  # type: ignore

                with tqdm(total=count) as pbar:
                    yield from get_rows(conn, pbar=lambda: pbar.update(1))
