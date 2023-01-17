import contextlib
import threading
from typing import Iterator, TypedDict

# FIXME add sqlalchemy stubs
import sqlalchemy as sa  # type: ignore

from system.namespace.load import NS_NAME_MAX_LEN
from system.namespace.module import MODULE_MAX_LEN
from system.namespace.namespace import ModuleName, Namespace


DBConfig = TypedDict('DBConfig', {
    "dialect": str,
    "host": str,
    "port": int,
    "user": str,
    "passwd": str,
    "dbname": str,
    "schema": str,
})


EngineKey = tuple[str, str, int, str, str]


def get_engine_key(config: DBConfig) -> EngineKey:
    return (
        config["dialect"],
        config["host"],
        config["port"],
        config["user"],
        config["dbname"],
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
        ENGINES[key] = res
    return res


class DBConnector:
    def __init__(self, config: DBConfig) -> None:
        self._engine = get_engine(config)
        self._schema = config["schema"]
        self._metadata_obj = sa.MetaData(self._engine, self._schema)
        self._namespaces: dict[str, int] = {}
        self._modules: dict[str, int] = {}

    def is_init(self) -> bool:
        if not self.get_table("namespace", autoload=False).exists():
            return False
        if not self.get_table("modules", autoload=False).exists():
            return False
        return True

    @contextlib.contextmanager
    def create_tables(self) -> Iterator[sa.MetaData]:
        metadata_obj = sa.MetaData(self._engine, self._schema)
        yield metadata_obj
        metadata_obj.create_all(checkfirst=True)

    @contextlib.contextmanager
    def create_module_tables(
            self,
            module: ModuleName,
            version: int) -> Iterator[tuple[sa.MetaData, sa.Column]]:
        current_version = self.get_module_version(module)
        if current_version == version:
            return
        if current_version != 0:
            raise ValueError(
                f"cannot upgrade from version {current_version} to {version}")
        with self._engine.begin() as conn:
            metadata_obj = sa.MetaData(conn, self._schema)
            yield metadata_obj, self.ns_key_column()
            self._set_module_version(conn, module, version)
            metadata_obj.create_all(checkfirst=True)

    def init_db(self) -> None:
        with self.create_tables() as metadata_obj:
            sa.Table(
                "namespace",
                metadata_obj,
                sa.Column(
                    "id",
                    sa.Integer,
                    primary_key=True,
                    autoincrement=True,
                    nullable=False,
                    unique=True),
                sa.Column(
                    "name",
                    sa.String(NS_NAME_MAX_LEN),
                    primary_key=True,
                    nullable=False,
                    unique=True))
            sa.Table(
                "modules",
                metadata_obj,
                sa.Column(
                    "module",
                    sa.String(MODULE_MAX_LEN),
                    primary_key=True,
                    nullable=False,
                    unique=True),
                sa.Column(
                    "version",
                    sa.Integer,
                    nullable=False))

    def get_table(self, name: str, *, autoload: bool = True) -> sa.Table:
        return sa.Table(name, self._metadata_obj, autoload=autoload)

    def _refresh_modules(self) -> None:
        with self.get_connection() as conn:
            t_modules = self.get_table("modules")
            res = conn.execute(sa.select(
                [t_modules.c.module, t_modules.c.version]))
            self._modules = {
                row.module: row.version
                for row in res
            }

    def get_module_version(self, module: ModuleName) -> int:
        res = self._modules.get(module)
        if res is None:
            self._refresh_modules()
            res = self._modules.get(module)
        if res is None:
            return 0
        return res

    def _set_module_version(
            self,
            conn: sa.engine.Connection,
            module: ModuleName,
            version: int) -> None:
        t_modules = self.get_table("modules")
        stmt = t_modules.insert().values(module=module, version=version)
        stmt = stmt.on_conflict_do_update(
            index_elements=[t_modules.c.module],
            index_where=t_modules.c.module.like(module),
            set_=dict(data=stmt.excluded.data))
        conn.execute(stmt)
        self._refresh_modules()

    def _refresh_namespaces(self) -> None:
        with self.get_connection() as conn:
            t_namespace = self.get_table("namespace")
            res = conn.execute(sa.select(
                [t_namespace.c.name, t_namespace.c.id]))
            self._namespaces = {
                row.name: row.id
                for row in res
            }

    def _add_namespace(self, ns_name: str) -> None:
        with self.get_connection() as conn:
            t_namespace = self.get_table("namespace")
            conn.execute(t_namespace.insert().values(name=ns_name))
        self._refresh_namespaces()

    def get_namespace_id(self, namespace: Namespace, *, create: bool) -> int:
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

    def ns_key_column(self) -> sa.Column:
        t_namespace = self.get_table("namespace")
        return t_namespace.c.id

    def get_schema(self) -> str:
        return self._schema

    def get_engine(self) -> sa.engine.Engine:
        return self._engine

    @contextlib.contextmanager
    def get_connection(self) -> Iterator[sa.engine.Connection]:
        with self._engine.connect() as conn:
            yield conn
