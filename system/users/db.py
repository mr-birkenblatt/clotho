from typing import Iterable

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import insert as pg_insert

from db.base import Base, NamespaceTable
from db.db import DBConnector
from system.namespace.namespace import Namespace
from system.users.store import UserStore
from system.users.user import MAX_USER_NAME_LEN, User


MODULE_VERSION = 1


class UsersTable(Base):  # pylint: disable=too-few-public-methods
    __tablename__ = "users"

    namespace_id = sa.Column(
        sa.Integer,
        sa.ForeignKey(
            NamespaceTable.id, onupdate="CASCADE", ondelete="CASCADE"),
        primary_key=True)
    id = sa.Column(
        sa.String(UserStore.id_length()),
        primary_key=True,
        nullable=False)
    name = sa.Column(
        sa.String(MAX_USER_NAME_LEN),
        nullable=False)
    data = sa.Column(sa.JSON(), nullable=False)


class DBUserStore(UserStore):
    def __init__(
            self,
            namespace: Namespace,
            db: DBConnector) -> None:
        super().__init__()
        self._namespace = namespace
        self._db = db
        self._nid: int | None = None

    def _get_nid(self) -> int:
        nid = self._nid
        if nid is None:
            nid = self._db.get_namespace_id(self._namespace, create=True)
            self._nid = nid
        return nid

    def _get_ns_name(self) -> str:
        return self._namespace.get_name()

    def is_module_init(self) -> bool:
        return self._db.is_module_init(self, MODULE_VERSION)

    def initialize_module(self) -> None:
        self._db.create_module_tables(
            self, MODULE_VERSION, [UsersTable])

    def get_user_by_id(self, user_id: str) -> User:
        with self._db.get_connection() as conn:
            stmt = sa.select([UsersTable.name, UsersTable.data]).where(sa.and_(
                UsersTable.namespace_id == self._get_nid(),
                UsersTable.id == user_id))
            res = conn.execute(stmt).one_or_none()
            if res is None:
                raise KeyError(f"user {user_id} does not exist")
            obj = res.data
            return User(res.name, obj["permissions"])

    def store_user(self, user: User) -> None:
        nid = self._get_nid()
        user_id = user.get_id()
        name = user.get_name()
        obj = {
            "permissions": user.get_permissions(),
        }
        with self._db.get_connection() as conn:
            values = {
                "namespace_id": nid,
                "id": user_id,
                "name": name,
                "data": obj,
            }
            stmt = pg_insert(UsersTable).values(values)
            stmt = stmt.on_conflict_do_update(
                index_elements=[UsersTable.namespace_id, UsersTable.id],
                index_where=sa.and_(
                    UsersTable.namespace_id == nid,
                    UsersTable.id == user_id),
                set_=dict(stmt.excluded.items()))
            conn.execute(stmt)

    def get_all_users(self) -> Iterable[User]:
        with self._db.get_connection() as conn:
            stmt = sa.select([UsersTable.name, UsersTable.data]).where(
                UsersTable.namespace_id == self._get_nid())
            for row in conn.execute(stmt):
                obj = row.data
                yield User(row.name, obj["permissions"])
