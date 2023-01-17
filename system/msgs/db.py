# FIXME add sqlalchemy stubs
import sqlalchemy as sa  # type: ignore

from db.db import DBConnector
from misc.lru import LRU
from system.msgs.message import Message, MHash
from system.msgs.store import MessageStore


MODULE_VERSION = 1
RELOAD_TOPICS_FREQ = 60 * 60  # 1h


class DBStore(MessageStore):
    def __init__(self, db: DBConnector, cache_size: int) -> None:
        super().__init__()
        self._db = db
        self._cache: LRU[MHash, Message] = LRU(cache_size)
        self._topics: list[Message] | None = None

    def is_module_init(self) -> bool:
        return self._db.table_exists("msgs")

    def initialize_module(self) -> None:
        self._db.init_db()
        with self._db.create_module_tables(
                self.module_name(), MODULE_VERSION) as (metadata_obj, ns_col):
            sa.Table(
                "msgs",
                metadata_obj,
                sa.Column(
                    "namespace_id",
                    None,
                    sa.ForeignKey(ns_col),
                    primary_key=True),
                sa.Column(
                    "mhash",
                    sa.String(MHash.parse_size()),
                    primary_key=True,
                    nullable=False),
                sa.Column("text", sa.Text, nullable=False))
            sa.Table(
                "topics",
                metadata_obj,
                sa.Column(
                    "namespace_id",
                    None,
                    sa.ForeignKey(ns_col),
                    primary_key=True),
                sa.Column(
                    "id",
                    sa.Integer,
                    primary_key=True,
                    autoincrement=True,
                    nullable=False,
                    unique=True),
                sa.Column(
                    "mhash",
                    sa.String(MHash.parse_size()),
                    nullable=False),
                sa.Column("topic", sa.Text, nullable=False))
