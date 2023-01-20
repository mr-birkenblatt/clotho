import re
from typing import Any, cast, get_args, Literal, TypedDict

from db.db import DBConfig
from model.embedding import EmbeddingProviderModule
from system.embedding.store import EmbedModule
from system.links.store import LinkModule
from system.msgs.store import MsgsModule
from system.suggest.suggest import SuggestModule
from system.users.store import UsersModule


RedisConfigObj = TypedDict('RedisConfigObj', {
    "host": str,
    "port": int,
    "passwd": str,
    "prefix": str,
    "path": str,
})
ConnectionObj = TypedDict('ConnectionObj', {
    "redis": dict[str, RedisConfigObj],
    "db": dict[str, DBConfig],
})
NamespaceObj = TypedDict('NamespaceObj', {
    "msgs": MsgsModule,
    "links": LinkModule,
    "suggest": list[SuggestModule],
    "users": UsersModule,
    "embed": EmbedModule,
    "model": EmbeddingProviderModule,
    "connections": ConnectionObj,
    "writeback": bool,
})


def redis_from_obj(
        ns_name: str,
        redis_obj: dict[str, Any]) -> dict[str, RedisConfigObj]:
    return {
        name: {
            "host": obj.get("host", "localhost"),
            "port": int(obj.get("port", 6379)),
            "passwd": obj.get("passwd", ""),
            "prefix": obj.get("prefix", f"{ns_name}"),
            "path": obj["path"],
        }
        for name, obj in redis_obj.items()
    }


def db_from_obj(db_obj: dict[str, Any]) -> dict[str, DBConfig]:
    return {
        name: {
            "dialect": obj.get("dialect", "postgresql"),
            "host": obj.get("host", "localhost"),
            "port": int(obj.get("port", 5432)),
            "user": obj["user"],
            "passwd": obj.get("passwd", ""),
            "dbname": obj["dbname"],
            "schema": obj.get("schema", "public"),
        }
        for name, obj in db_obj.items()
    }


def msgs_from_obj(obj: dict[str, Any]) -> MsgsModule:
    res: MsgsModule
    name: str = obj.get("name", "disk")
    if name == "ram":
        res = {
            "name": "ram",
        }
    elif name == "disk":
        res = {
            "name": "disk",
            "cache_size": obj.get("cache_size", 50000),
        }
    elif name == "cold":
        res = {
            "name": "cold",
            "keep_alive": obj.get("keep_alive", 1.0),
        }
    elif name == "db":
        res = {
            "name": "db",
            "conn": obj["conn"],
            "cache_size": obj.get("cache_size", 1000),
        }
    else:
        raise ValueError(f"invalid name {name} {obj}")
    return res


def links_from_obj(obj: dict[str, Any]) -> LinkModule:
    res: LinkModule
    name: str = obj.get("name", "redis")
    if name == "redis":
        res = {
            "name": "redis",
            "conn": obj["conn"],
        }
    else:
        raise ValueError(f"invalid name {name} {obj}")
    return res


def suggest_from_obj(obj: dict[str, Any]) -> SuggestModule:
    res: SuggestModule
    name: str = obj.get("name", "random")
    if name == "random":
        res = {
            "name": "random",
        }
    elif name == "model":
        res = {
            "name": "model",
            "count": 10,
        }
    else:
        raise ValueError(f"invalid name {name} {obj}")
    return res


def users_from_obj(obj: dict[str, Any]) -> UsersModule:
    res: UsersModule
    name: str = obj.get("name", "disk")
    if name == "ram":
        res = {
            "name": "ram",
        }
    elif name == "cold":
        res = {
            "name": "cold",
            "keep_alive": obj.get("keep_alive", 1.0),
        }
    elif name == "disk":
        res = {
            "name": "disk",
            "cache_size": obj.get("cache_size", 50000),
        }
    else:
        raise ValueError(f"invalid name {name} {obj}")
    return res


CacheEmbedName = Literal["redis", "db"]
CACHE_EMBED_NAMES = get_args(CacheEmbedName)


def embed_from_obj(obj: dict[str, Any]) -> EmbedModule:
    res: EmbedModule
    name: str = obj.get("name", "none")
    if name == "none":
        res = {
            "name": "none",
        }
    elif name in CACHE_EMBED_NAMES:
        res = {
            "name": cast(CacheEmbedName, name),
            "conn": obj["conn"],
            "path": obj.get("path", "embed"),
            "index": obj["index"],
            "trees": obj.get("trees", 100),
            "shard_size": obj.get("shard_size", 100000),
            "metric": obj.get("metric", "dot"),
        }
    else:
        raise ValueError(f"invalid name {name} {obj}")
    return res


def model_from_obj(obj: dict[str, Any]) -> EmbeddingProviderModule:
    res: EmbeddingProviderModule
    name: str = obj.get("name", "none")
    if name == "none":
        res = {
            "name": "none",
        }
    elif name == "transformer":
        res = {
            "name": "transformer",
            "fname": obj["fname"],
            "version": int(obj["version"]),
            "is_harness": bool(obj["is_harness"]),
        }
    else:
        raise ValueError(f"invalid name {name} {obj}")
    return res


NS_NAME_MAX_LEN = 40
VALID_NS_NAME = re.compile(r"^[a-z][a-z0-9_-]+$")


def ns_from_obj(ns_name: str, obj: dict[str, Any]) -> NamespaceObj:
    if len(ns_name) > NS_NAME_MAX_LEN or VALID_NS_NAME.search(ns_name) is None:
        raise ValueError(f"invalid namespace name {ns_name}")
    conns = obj.get("connections", {})
    return {
        "msgs": msgs_from_obj(obj.get("msgs", {})),
        "links": links_from_obj(obj.get("links", {})),
        "suggest": [
            suggest_from_obj(cur)
            for cur in obj.get("suggest", [])
        ],
        "users": users_from_obj(obj.get("users", {})),
        "embed": embed_from_obj(obj.get("embed", {})),
        "model": model_from_obj(obj.get("model", {})),
        "connections": {
            "redis": redis_from_obj(ns_name, conns.get("redis", {})),
            "db": db_from_obj(conns.get("db", {})),
        },
        "writeback": obj.get("writeback", True),
    }
