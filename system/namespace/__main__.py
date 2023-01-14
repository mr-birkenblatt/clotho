
import argparse
import os
import sys
from typing import TypedDict

from misc.io import ensure_folder
from misc.redis import get_redis_config
from system.namespace.namespace import Namespace
from system.namespace.store import get_namespace


Module = TypedDict('Module', {
    "name": str,
    "port": int,
    "path": str,
})


def get_module(namespace: Namespace, module: str) -> Module:
    if module == "link":
        lmodule = namespace.get_link_module()
        if lmodule["name"] != "redis":
            raise ValueError(f"incompatible module {lmodule}")
        return {
            "name": lmodule["name"],
            "port": lmodule["port"],
            "path": os.path.join(namespace.get_root(), lmodule["path"]),
        }
    if module == "embed":
        emodule = namespace.get_embed_module()
        if emodule["name"] != "redis":
            raise ValueError(f"incompatible module {emodule}")
        return {
            "name": emodule["name"],
            "port": emodule["port"],
            "path": os.path.join(namespace.get_root(), emodule["path"]),
        }
    raise ValueError(f"invalid module: {module}")


def get_port(ns_name: str, module: str) -> int:
    # FIXME: eventually load up the full redis config and maybe start it
    if ns_name.startswith("_"):
        cfg = get_redis_config((ns_name, ""))
        return cfg["port"]
    namespace = get_namespace(ns_name)
    module_obj = get_module(namespace, module)
    if module_obj["name"] != "redis":
        raise ValueError(f"no redis needed for module: {module_obj}")
    return module_obj["port"]


def get_path(ns_name: str, module: str) -> str:
    if ns_name.startswith("_"):
        cfg = get_redis_config((ns_name, ""))
        return cfg["path"]
    namespace = get_namespace(ns_name)
    module_obj = get_module(namespace, module)
    if module_obj["name"] != "redis":
        raise ValueError(f"no redis needed for module: {module_obj}")
    return module_obj["path"]


def run() -> None:
    stdout = sys.stdout
    args = parse_args()
    if args.info == "port":
        stdout.write(f"{get_port(args.namespace, args.module)}")
        stdout.flush()
    elif args.info == "path":
        path = ensure_folder(get_path(args.namespace, args.module))
        stdout.write(f"{path}")
        stdout.flush()
    else:
        raise RuntimeError(f"invalid info: {args.info}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog=f"python -m {os.path.basename(os.path.dirname(__file__))}",
        description="Extract namespace information")
    parser.add_argument(
        "info",
        choices=["port", "path"],
        help="what information to extract")
    parser.add_argument("--namespace", default="default", help="the namespace")
    parser.add_argument("--module", default="link", help="the module")
    return parser.parse_args()


if __name__ == "__main__":
    run()
