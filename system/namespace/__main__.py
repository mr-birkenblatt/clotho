
import argparse
import os
import sys
from typing import TypedDict

from misc.io import ensure_folder
from misc.redis import get_redis_config
from system.namespace.module import get_module_obj
from system.namespace.namespace import (
    get_module_name,
    MODULE_EMBED,
    MODULE_LINKS,
    ModuleName,
    Namespace,
)
from system.namespace.store import get_namespace


Module = TypedDict('Module', {
    "name": str,
    "port": int,
    "path": str,
})


def get_module(namespace: Namespace, module: ModuleName) -> Module:
    if module == MODULE_LINKS:
        lmodule = namespace.get_link_module()
        if lmodule["name"] != "redis":
            raise ValueError(f"incompatible module {lmodule}")
        config = namespace.get_redis_config(lmodule["conn"])
        return {
            "name": lmodule["name"],
            "port": config["port"],
            "path": os.path.join(namespace.get_root(), config["path"]),
        }
    if module == MODULE_EMBED:
        emodule = namespace.get_embed_module()
        if emodule["name"] != "redis":
            raise ValueError(f"incompatible module {emodule}")
        config = namespace.get_redis_config(emodule["conn"])
        return {
            "name": emodule["name"],
            "port": config["port"],
            "path": os.path.join(namespace.get_root(), config["path"]),
        }
    raise ValueError(f"invalid module: {module}")


def get_port(ns_name: str, module: ModuleName) -> int:
    # FIXME: eventually load up the full redis config and maybe start it
    if ns_name.startswith("_"):
        cfg = get_redis_config((ns_name, ""))
        return cfg["port"]
    namespace = get_namespace(ns_name)
    module_obj = get_module(namespace, module)
    if module_obj["name"] != "redis":
        raise ValueError(f"no redis needed for module: {module_obj}")
    return module_obj["port"]


def get_path(ns_name: str, module: ModuleName) -> str:
    if ns_name.startswith("_"):
        cfg = get_redis_config((ns_name, ""))
        return cfg["path"]
    namespace = get_namespace(ns_name)
    module_obj = get_module(namespace, module)
    if module_obj["name"] != "redis":
        raise ValueError(f"no redis needed for module: {module_obj}")
    return module_obj["path"]


def init(ns_name: str, module: ModuleName) -> None:
    namespace = get_namespace(ns_name)
    module_obj = get_module_obj(namespace, module)
    if module_obj.is_module_init():
        print(
            f"module {module} already initialized "
            f"(ns: {namespace.get_name()})")
        return
    module_obj.initialize_module()


def xfer(ns_name: str, module: ModuleName, ns_dest: str) -> None:
    namespace = get_namespace(ns_name)
    dest_namespace = get_namespace(ns_dest)
    module_obj = get_module_obj(dest_namespace, module)
    module_obj.ensure_module_init(ns_name)
    module_obj.from_namespace(namespace, progress_bar=True)


def run() -> None:
    stdout = sys.stdout
    args = parse_args()
    if args.dest is not None and args.command != "xfer":
        raise RuntimeError("--dest set for command other than 'xfer'")
    ns_name: str = args.namespace
    module = get_module_name(args.module)
    if args.command == "port":
        stdout.write(f"{get_port(ns_name, module)}")
        stdout.flush()
    elif args.command == "path":
        path = ensure_folder(get_path(ns_name, module))
        stdout.write(f"{path}")
        stdout.flush()
    elif args.command == "init":
        init(ns_name, module)
    elif args.command == "xfer":
        xfer(ns_name, module, args.dest)
    else:
        raise RuntimeError(f"invalid command: {args.command}")


def parse_args() -> argparse.Namespace:
    from system.namespace import NAMESPACE_EXEC

    parser = argparse.ArgumentParser(
        prog=NAMESPACE_EXEC,
        description="Extract namespace information")
    parser.add_argument(
        "command",
        choices=["port", "path", "init", "xfer"],
        help="command or what information to extract")
    parser.add_argument("--namespace", default="default", help="the namespace")
    parser.add_argument("--module", default="link", help="the module")
    parser.add_argument(
        "--dest", default=None, help="the destination namespace for 'xfer'")
    return parser.parse_args()


if __name__ == "__main__":
    run()
