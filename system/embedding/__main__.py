import argparse
import os
from typing import Literal

from system.embedding.store import get_embed_store
from system.msgs.message import Message
from system.msgs.store import get_message_store
from system.namespace.store import get_namespace


def run() -> None:
    args = parse_args()
    namespace = get_namespace(args.namespace)
    msg_store = get_message_store(namespace)
    embed_store = get_embed_store(namespace)
    if args.text is not None:
        name_from: Literal["parent"] = "parent"
        name_to: Literal["child"] = "child"
        print(f"from: {name_from} to: {name_to}")
        msg = Message(msg=args.text)
        mhash = msg_store.write_message(msg)
        embed = embed_store.get_embedding(msg_store, name_from, mhash)
        divide = "=" * 42
        print(f"query: {msg.get_text()}")
        for out in embed_store.get_closest(name_to, embed, 20):
            print(divide)
            print(msg_store.read_message(out).get_text())
    else:
        embed_store.ensure_all(msg_store)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog=f"python -m {os.path.basename(os.path.dirname(__file__))}",
        description="Precompute embeddings")
    parser.add_argument("--namespace", default="default", help="the namespace")
    parser.add_argument(
        "--text",
        default=None,
        help="optional text for retrieval. this prevents the full indexing")
    return parser.parse_args()


if __name__ == "__main__":
    run()
