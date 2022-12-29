import os

from misc.env import envload_path
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
                "host": cfg["host"],
                "port": cfg["port"],
                "passwd": cfg["passwd"],
                "prefix": cfg["prefix"],
                "path": cfg["path"],
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
        })
    return TEST_NAMESPACE


NS_CACHE: dict[str, Namespace] = {}


def get_namespace(ns_name: str) -> Namespace:
    res = NS_CACHE.get(ns_name)
    if res is None:
        base_path = envload_path("USER_PATH", default="userdata")
        fname = os.path.join(base_path, "namespace", f"{ns_name}.json")
        try:
            with open_read(fname, text=True) as fin:
                obj = json_load(fin)
        except FileNotFoundError:
            if ns_name != "default":
                raise
            obj = {}
        ns_obj = ns_from_obj(ns_name, obj)
        res = Namespace(ns_name, ns_obj)
        NS_CACHE[ns_name] = res
        if ns_name == "default":
            out_obj = json_pretty(ns_obj)
            if out_obj != json_pretty(obj):
                ensure_folder(os.path.dirname(fname))
                with open_write(fname, text=True) as fout:
                    fout.write(out_obj)
                    fout.write(NL)
    return res
