import contextlib
import sys
import threading
from typing import Iterator, Type, TYPE_CHECKING, TypedDict

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import insert as pg_insert

from db.base import MHashTable


if TYPE_CHECKING:
    from db.base import Base
    from system.namespace.module import ModuleBase
    from system.namespace.namespace import Namespace


DBConfig = TypedDict('DBConfig', {
    "dialect": str,
    "host": str,
    "port": int,
    "user": str,
    "passwd": str,
    "dbname": str,
    "schema": str,
})


EngineKey = tuple[str, str, int, str, str, str]


def get_engine_key(config: DBConfig) -> EngineKey:
    return (
        config["dialect"],
        config["host"],
        config["port"],
        config["user"],
        config["dbname"],
        config["schema"],
    )


LOCK = threading.RLock()
ENGINES: dict[EngineKey, sa.engine.Engine] = {}


def get_engine(config: DBConfig) -> sa.engine.Engine:
    key = get_engine_key(config)
    res = ENGINES.get(key)
    if res is not None:
        return res
    with LOCK:
        res = ENGINES.get(key)
        if res is not None:
            return res
        dialect = config["dialect"]
        if dialect != "postgresql":
            print(
                "dialects other than 'postgresql' are not supported. "
                "continue at your own risk", file=sys.stderr)
        user = config["user"]
        passwd = config["passwd"]
        host = config["host"]
        port = config["port"]
        dbname = config["dbname"]
        res = sa.create_engine(
            f"{dialect}://{user}:{passwd}@{host}:{port}/{dbname}")
        res = res.execution_options(
            schema_translate_map={None: config["schema"]})
        ENGINES[key] = res
    return res


class DBConnector:
    def __init__(self, config: DBConfig) -> None:
        self._engine = get_engine(config)
        self._namespaces: dict[str, int] = {}
        self._modules: dict[str, int] = {}

    def table_exists(self, table: Type['Base']) -> bool:
        return sa.inspect(self._engine).has_table(
            table.__table__.name)

    def create_tables(self, tables: list[Type['Base']]) -> None:
        from db.base import Base

        Base.metadata.create_all(
            self._engine,
            tables=[table.__table__ for table in tables],
            checkfirst=True)

    def is_init(self) -> bool:
        from db.base import ModulesTable, NamespaceTable

        if not self.table_exists(NamespaceTable):
            return False
        if not self.table_exists(ModulesTable):
            return False
        if not self.table_exists(MHashTable):
            return False
        return True

    def init_db(self) -> None:
        from db.base import ModulesTable, NamespaceTable

        if self.is_init():
            return
        self.create_tables([NamespaceTable, ModulesTable, MHashTable])

    def is_module_init(
            self,
            module: 'ModuleBase' | Type['ModuleBase'],
            version: int,
            submodule: str | None = None) -> bool:
        if not self.is_init():
            return False
        return self.get_module_version(module, submodule) == version

    def create_module_tables(
            self,
            module: 'ModuleBase' | Type['ModuleBase'],
            version: int,
            tables: list[Type['Base']],
            submodule: str | None = None,
            *,
            force: bool) -> None:
        if not self.is_init():
            self.init_db()
        current_version = self.get_module_version(module, submodule)
        if not force and current_version == version:
            return
        if not force and current_version != 0:
            raise ValueError(
                f"cannot upgrade from version {current_version} to {version}")
        self.create_tables(tables)
        self._set_module_version(module, submodule, version)

    def _refresh_modules(self) -> None:
        from db.base import ModulesTable

        with self.get_connection() as conn:
            stmt = sa.select([ModulesTable.module, ModulesTable.version])
            self._modules = {
                row.module: row.version
                for row in conn.execute(stmt)
            }

    def get_module_version(
            self,
            module: 'ModuleBase' | Type['ModuleBase'],
            submodule: str | None = None) -> int:
        module_name = module.module_name()
        if submodule is not None:
            name = f"{module_name}:{submodule}"
        else:
            name = module_name
        res = self._modules.get(name)
        if res is None:
            self._refresh_modules()
            res = self._modules.get(name)
        if res is None:
            return 0
        return res

    def _set_module_version(
            self,
            module: 'ModuleBase' | Type['ModuleBase'],
            submodule: str | None,
            version: int) -> None:
        from db.base import ModulesTable

        module_name = module.module_name()
        if submodule is not None:
            name = f"{module_name}:{submodule}"
        else:
            name = module_name
        with self.get_connection() as conn:
            values = {
                "module": name,
                "version": version,
            }
            stmt = pg_insert(ModulesTable).values(values)
            stmt = stmt.on_conflict_do_update(
                index_elements=[ModulesTable.module],
                index_where=ModulesTable.module.like(name),
                set_=dict(stmt.excluded.items()))
            conn.execute(stmt)
        self._refresh_modules()

    def _refresh_namespaces(self) -> None:
        from db.base import NamespaceTable

        with self.get_connection() as conn:
            stmt = sa.select(
                [NamespaceTable.name, NamespaceTable.id])
            self._namespaces = {
                row.name: row.id
                for row in conn.execute(stmt)
            }

    def _add_namespace(self, ns_name: str) -> None:
        from db.base import NamespaceTable

        with self.get_connection() as conn:
            stmt = pg_insert(NamespaceTable).values(name=ns_name)
            stmt = stmt.on_conflict_do_nothing()
            conn.execute(stmt)
        self._refresh_namespaces()

    def get_namespace_id(self, namespace: 'Namespace', *, create: bool) -> int:
        return self._get_namespace_id(namespace.get_name(), create=create)

    def _get_namespace_id(self, ns_name: str, *, create: bool) -> int:
        res = self._namespaces.get(ns_name)
        if res is None:
            self._refresh_namespaces()
            res = self._namespaces.get(ns_name)
        if res is None:
            if not create:
                raise KeyError(f"unknown namespace: {ns_name}")
            self._add_namespace(ns_name)
            res = self._namespaces.get(ns_name)
        if res is None:
            raise KeyError(f"cannot create namespace: {ns_name}")
        return res

    def get_engine(self) -> sa.engine.Engine:
        return self._engine

    @contextlib.contextmanager
    def get_connection(self) -> Iterator[sa.engine.Connection]:
        with self._engine.connect() as conn:
            yield conn

    @contextlib.contextmanager
    def get_session(self) -> Iterator[sa.orm.Session]:
        with sa.orm.Session(self._engine) as session:
            yield session
