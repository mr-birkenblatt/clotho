import os

from misc.env import envload_path
from misc.io import open_read
from misc.redis import get_test_config
from misc.util import json_load
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
            },
            "suggest": {
                "name": "random",
            },
            "users": {
                "name": "ram",
            },
        })
    return TEST_NAMESPACE


NS_CACHE: dict[str, Namespace] = {}


def get_namespace(ns_name: str) -> Namespace:
    res = NS_CACHE.get(ns_name)
    if res is None:
        base_path = envload_path("USER_PATH", default="userdata")
        fname = os.path.join(base_path, "namespace", f"{ns_name}.json")
        with open_read(fname, text=True) as fin:
            obj = json_load(fin)
        ns_obj = ns_from_obj(ns_name, obj)
        res = Namespace(ns_name, ns_obj)
        NS_CACHE[ns_name] = res
    return res
