import re
from typing import Any, TypedDict

from system.links.store import LinkModule
from system.msgs.store import MsgsModule
from system.suggest.suggest import SuggestModule
from system.users.store import UsersModule


NamespaceObj = TypedDict('NamespaceObj', {
    "msgs": MsgsModule,
    "links": LinkModule,
    "suggest": SuggestModule,
    "users": UsersModule,
})


def msgs_from_obj(ns_name: str, obj: dict[str, Any]) -> MsgsModule:
    res: MsgsModule
    name = obj.get("name", "disk")
    if name == "ram":
        res = {
            "name": "ram"
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
    name = obj.get("name", "disk")
    if name == "redis":
        res = {
            "name": "redis",
            "host": obj.get("host", "localhost"),
            "port": int(obj.get("port", 6379)),
            "passwd": obj.get("passwd", ""),
            "prefix": obj.get("prefix", f"{ns_name}")
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


VALID_NS_NAME = re.compile(r"^[a-z][a-z0-9_-]+$")


def ns_from_obj(ns_name: str, obj: dict[str, Any]) -> NamespaceObj:
    if VALID_NS_NAME.search(ns_name) is None:
        raise ValueError(f"invalid namespace name {ns_name}")
    return {
        "msgs": msgs_from_obj(ns_name, obj.get("msgs", {})),
        "links": links_from_obj(ns_name, obj.get("links", {})),
        "suggest": suggest_from_obj(obj.get("suggest", {})),
        "users": users_from_obj(ns_name, obj.get("users", {})),
    }
