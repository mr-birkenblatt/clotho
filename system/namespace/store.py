import os

from misc.io import ensure_folder, open_read, open_write
from misc.redis import get_test_config
from misc.util import json_load, json_pretty, NL
from system.namespace.load import ns_from_obj
from system.namespace.namespace import Namespace


TEST_NAMESPACE_NAME = "_test"
TEST_NAMESPACE: Namespace | None = None


def get_test_namespace() -> Namespace:
    global TEST_NAMESPACE

    if TEST_NAMESPACE is None:
        cfg = get_test_config()
        TEST_NAMESPACE = Namespace(TEST_NAMESPACE_NAME, {
            "msgs": {
                "name": "ram",
            },
            "links": {
                "name": "redis",
                "conn": "links",
            },
            "suggest": [
                {
                    "name": "random",
                },
            ],
            "users": {
                "name": "ram",
            },
            "embed": {
                "name": "none",
            },
            "model": {
                "name": "none",
            },
            "connections": {
                "redis": {
                    "links": {
                        "host": cfg["host"],
                        "port": cfg["port"],
                        "passwd": cfg["passwd"],
                        "prefix": cfg["prefix"],
                        "path": cfg["path"],
                    },
                },
            },
            "writeback": False,
        })
    return TEST_NAMESPACE


NS_CACHE: dict[str, Namespace] = {}


def get_namespace(ns_name: str) -> Namespace:
    res = NS_CACHE.get(ns_name)
    if res is None:
        root = Namespace.get_root_for(ns_name)
        fname = os.path.join(root, "settings.json")
        try:
            with open_read(fname, text=True) as fin:
                obj = json_load(fin)
        except FileNotFoundError:
            if ns_name != "default":
                raise
            obj = {}
        ns_obj = ns_from_obj(ns_name, obj)
        is_writeback = ns_obj.get("writeback", True)
        ns_obj["writeback"] = False
        res = Namespace(ns_name, ns_obj)
        NS_CACHE[ns_name] = res
        if is_writeback:
            out_obj = json_pretty(ns_obj)
            if out_obj != json_pretty(obj):
                ensure_folder(root)
                with open_write(fname, text=True) as fout:
                    fout.write(out_obj)
                    fout.write(NL)
    return res
