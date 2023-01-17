import contextlib
import threading
from typing import Iterator, Type, TYPE_CHECKING, TypedDict

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import insert as pg_insert


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
            getattr(table, "__table__").name)

    def create_tables(self, tables: list[Type['Base']]) -> None:
        from db.base import Base

        metadata: sa.MetaData = Base.metadata
        metadata.create_all(
            self._engine,
            tables=[getattr(table, "__table__") for table in tables],
            checkfirst=True)

    def is_init(self) -> bool:
        from db.base import ModulesTable, NamespaceTable

        if not self.table_exists(NamespaceTable):
            return False
        if not self.table_exists(ModulesTable):
            return False
        return True

    def init_db(self) -> None:
        from db.base import ModulesTable, NamespaceTable

        if self.is_init():
            return
        self.create_tables([NamespaceTable, ModulesTable])

    def is_module_init(self, module: 'ModuleBase', version: int) -> bool:
        if not self.is_init():
            return False
        return self.get_module_version(module) == version

    def create_module_tables(
            self,
            module: 'ModuleBase',
            version: int,
            tables: list[Type['Base']]) -> None:
        if not self.is_init():
            self.init_db()
        current_version = self.get_module_version(module)
        if current_version == version:
            return
        if current_version != 0:
            raise ValueError(
                f"cannot upgrade from version {current_version} to {version}")
        self.create_tables(tables)
        self._set_module_version(module, version)

    def _refresh_modules(self) -> None:
        from db.base import ModulesTable

        with self.get_connection() as conn:
            stmt = sa.select(
                [ModulesTable.module, ModulesTable.version])
            self._modules = {
                row.module: row.version
                for row in conn.execute(stmt)
            }

    def get_module_version(self, module: 'ModuleBase') -> int:
        module_name = module.module_name()
        res = self._modules.get(module_name)
        if res is None:
            self._refresh_modules()
            res = self._modules.get(module_name)
        if res is None:
            return 0
        return res

    def _set_module_version(
            self,
            module: 'ModuleBase',
            version: int) -> None:
        from db.base import ModulesTable

        module_name = module.module_name()
        with self.get_connection() as conn:
            values = {
                "module": module_name,
                "version": version
            }
            stmt = pg_insert(ModulesTable).values(values)
            stmt = stmt.on_conflict_do_update(
                index_elements=[ModulesTable.module],
                index_where=ModulesTable.module.like(module_name),
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
            stmt = stmt.on_conflict_do_nothin()
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
