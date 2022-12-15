
import argparse
import os
import sys

from misc.env import envload_path
from misc.io import ensure_folder
from misc.redis import get_redis_config
from system.namespace.store import get_namespace


def get_port(ns_name: str) -> int:
    # FIXME: eventually load up the full redis config and maybe start it
    if ns_name.startswith("_"):
        cfg = get_redis_config((ns_name, ""))
        return cfg["port"]
    namespace = get_namespace(ns_name)
    link_module = namespace.get_link_module()
    if link_module["name"] != "redis":
        raise ValueError(f"no redis needed for link module: {link_module}")
    return link_module["port"]


def get_path(ns_name: str) -> str:
    if ns_name.startswith("_"):
        cfg = get_redis_config((ns_name, ""))
        return cfg["path"]
    namespace = get_namespace(ns_name)
    link_module = namespace.get_link_module()
    if link_module["name"] != "redis":
        raise ValueError(f"no redis needed for link module: {link_module}")
    base_path = envload_path("USER_PATH", default="userdata")
    return os.path.join(base_path, link_module["path"])


def run() -> None:
    stdout = sys.stdout
    args = parse_args()
    if args.info == "port":
        stdout.write(f"{get_port(args.namespace)}")
        stdout.flush()
    elif args.info == "path":
        path = ensure_folder(get_path(args.namespace))
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
    return parser.parse_args()


if __name__ == "__main__":
    run()
