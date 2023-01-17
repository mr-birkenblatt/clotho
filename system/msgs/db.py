from typing import Iterable

import numpy as np
import sqlalchemy as sa

from db.base import Base, NamespaceTable
from db.db import DBConnector
from misc.lru import LRU
from system.msgs.message import Message, MHash
from system.msgs.store import MessageStore


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
        nullable=False,
        unique=True)
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
        nullable=False,
        unique=True)
    mhash = sa.Column(
        sa.String(MHash.parse_size()),
        nullable=False,
        unique=True)
    topic = sa.Column(sa.Text, nullable=False)


class DBStore(MessageStore):
    def __init__(self, db: DBConnector, cache_size: int) -> None:
        super().__init__()
        self._db = db
        self._cache: LRU[MHash, Message] = LRU(cache_size)
        self._topics: list[Message] | None = None

    def is_module_init(self) -> bool:
        return self._db.is_module_init(self, MODULE_VERSION)

    def initialize_module(self) -> None:
        self._db.create_module_tables(
            self, MODULE_VERSION, [MsgsTable, TopicsTable])

    def write_message(self, message: Message) -> MHash:
        raise NotImplementedError()

    def read_message(self, message_hash: MHash) -> Message:
        raise NotImplementedError()

    def add_topic(self, topic: Message) -> MHash:
        raise NotImplementedError()

    def get_topics(
            self,
            offset: int,
            limit: int | None) -> list[Message]:
        raise NotImplementedError()

    def get_topics_count(self) -> int:
        return len(self.get_topics(0, None))

    def do_get_random_messages(
            self, rng: np.random.Generator, count: int) -> Iterable[MHash]:
        raise NotImplementedError()

    def enumerate_messages(self, *, progress_bar: bool) -> Iterable[MHash]:
        raise NotImplementedError()
