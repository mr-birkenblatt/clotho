import argparse
import os

from misc.util import python_module
from system.embedding.dbcache import register_model
from system.namespace.store import get_namespace


def parse_args() -> argparse.Namespace:
    description = (
        "Registers a model in the database.")
    parser = argparse.ArgumentParser(
        prog=f"python -m {python_module()}",
        description=description)
    parser.add_argument("--namespace", help="the namespace")
    parser.add_argument("--connection", help="the db connection")
    parser.add_argument("--file", help="the file the model is stored in")
    parser.add_argument("--version", type=int, help="the model version")
    parser.add_argument(
        "--is-harness",
        default=False,
        action="store_true",
        help="whether the model file is the harness or the actual model")
    return parser.parse_args()


def run() -> None:
    args = parse_args()
    ns_name = args.namespace
    conn = args.connection
    fname = os.path.abspath(args.file)
    root = os.path.dirname(fname)
    base = os.path.basename(fname)
    version = args.version
    is_harness = args.is_harness
    namespace = get_namespace(ns_name)
    db = namespace.get_db_connector(conn)
    model_hash = register_model(db, root, base, version, is_harness)
    model_name = base
    rix = model_name.rfind(".")
    if rix >= 0:
        model_name = model_name[:rix]
    print(f"model {model_name} registered as {model_hash}")


if __name__ == "__main__":
    run()
