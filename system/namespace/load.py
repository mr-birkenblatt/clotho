import re
from typing import Any, TypedDict

from model.embedding import EmbeddingProviderModule
from system.embedding.store import EmbedModule
from system.links.store import LinkModule
from system.msgs.store import MsgsModule
from system.suggest.suggest import SuggestModule
from system.users.store import UsersModule


NamespaceObj = TypedDict('NamespaceObj', {
    "msgs": MsgsModule,
    "links": LinkModule,
    "suggest": list[SuggestModule],
    "users": UsersModule,
    "embed": EmbedModule,
    "model": EmbeddingProviderModule,
})


def msgs_from_obj(ns_name: str, obj: dict[str, Any]) -> MsgsModule:
    res: MsgsModule
    name = obj.get("name", "disk")
    if name == "ram":
        res = {
            "name": "ram",
        }
    elif name == "disk":
        res = {
            "name": "disk",
            "root": obj.get("root", f"{ns_name}"),
        }
    else:
        raise ValueError(f"invalid name {name} {obj}")
    return res


def links_from_obj(ns_name: str, obj: dict[str, Any]) -> LinkModule:
    res: LinkModule
    name = obj.get("name", "redis")
    if name == "redis":
        res = {
            "name": "redis",
            "host": obj.get("host", "localhost"),
            "port": int(obj.get("port", 6379)),
            "passwd": obj.get("passwd", ""),
            "prefix": obj.get("prefix", f"{ns_name}"),
            "path": obj.get("path", f"{ns_name}"),
        }
    else:
        raise ValueError(f"invalid name {name} {obj}")
    return res


def suggest_from_obj(obj: dict[str, Any]) -> SuggestModule:
    res: SuggestModule
    name = obj.get("name", "random")
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


def users_from_obj(ns_name: str, obj: dict[str, Any]) -> UsersModule:
    res: MsgsModule
    name = obj.get("name", "disk")
    if name == "ram":
        res = {
            "name": "ram",
        }
    elif name == "disk":
        res = {
            "name": "disk",
            "root": obj.get("root", f"{ns_name}"),
        }
    else:
        raise ValueError(f"invalid name {name} {obj}")
    return res


def embed_from_obj(ns_name: str, obj: dict[str, Any]) -> EmbedModule:
    res: EmbedModule
    name = obj.get("name", "none")
    if name == "none":
        res = {
            "name": "none",
        }
    elif name == "redis":
        res = {
            "name": "redis",
            "host": obj.get("host", "localhost"),
            "port": int(obj.get("port", 6379)),
            "passwd": obj.get("passwd", ""),
            "prefix": obj.get("prefix", f"{ns_name}"),
            "path": obj.get("path", f"{ns_name}"),
            "index": obj["index"],
            "trees": obj.get("trees", 1000),
        }
    else:
        raise ValueError(f"invalid name {name} {obj}")
    return res


def model_from_obj(obj: dict[str, Any]) -> EmbeddingProviderModule:
    res: EmbeddingProviderModule
    name = obj.get("name", "none")
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


VALID_NS_NAME = re.compile(r"^[a-z][a-z0-9_-]+$")


def ns_from_obj(ns_name: str, obj: dict[str, Any]) -> NamespaceObj:
    if VALID_NS_NAME.search(ns_name) is None:
        raise ValueError(f"invalid namespace name {ns_name}")
    return {
        "msgs": msgs_from_obj(ns_name, obj.get("msgs", {})),
        "links": links_from_obj(ns_name, obj.get("links", {})),
        "suggest": [
            suggest_from_obj(cur)
            for cur in obj.get("suggest", [])
        ],
        "users": users_from_obj(ns_name, obj.get("users", {})),
        "embed": embed_from_obj(ns_name, obj.get("embed", {})),
        "model": model_from_obj(obj.get("model", {})),
    }
