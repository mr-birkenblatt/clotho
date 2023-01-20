import argparse
from typing import Literal

from misc.util import python_module
from system.embedding.store import get_embed_store
from system.msgs.message import Message
from system.msgs.store import get_message_store
from system.namespace.store import get_namespace


def run() -> None:
    args = parse_args()
    namespace = get_namespace(args.namespace)
    embed_store = get_embed_store(namespace)
    if args.self_test:
        embed_store.self_test("child", None)
        return
    msg_store = get_message_store(namespace)
    if args.text is not None:
        name_from: Literal["parent"] = "parent"
        name_to: Literal["child"] = "child"
        print(f"from: {name_from} to: {name_to}")
        msg = Message(msg=args.text)
        mhash = msg_store.write_message(msg)
        embed = embed_store.get_embedding(msg_store, name_from, mhash)
        divide = "=" * 42
        print(f"query: {msg.get_text()}")
        for out in embed_store.get_closest(
                name_to, embed, 20, precise=args.precise):
            if args.count:
                continue
            print(divide)
            print(msg_store.read_message(out).get_text())
    elif args.count:
        count = 0
        for _ in msg_store.enumerate_messages(progress_bar=False):
            count += 1
        print(f"{count} messages available")
    else:
        embed_store.ensure_all(msg_store)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog=f"python -m {python_module()}",
        description="Precompute embeddings")
    parser.add_argument("--namespace", default="default", help="the namespace")
    parser.add_argument(
        "--text",
        default=None,
        help="optional text for retrieval. this prevents the full indexing")
    parser.add_argument(
        "--precise",
        default=False,
        action="store_true",
        help="whether index lookup should be precise (only used with --text)")
    parser.add_argument(
        "--count",
        default=False,
        action="store_true",
        help="just count and don't do anything else")
    parser.add_argument(
        "--self-test",
        default=False,
        action="store_true",
        help="check whether distance definitions are compatible")
    return parser.parse_args()


if __name__ == "__main__":
    run()
