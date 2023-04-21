import argparse

from misc.util import python_module
from model.embedding import ProviderRole
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
    precise: bool = args.precise
    no_cache: bool = args.no_cache
    if args.text is not None:
        is_flip: bool = args.parent
        name_from: ProviderRole = "child" if is_flip else "parent"
        name_to: ProviderRole = "parent" if is_flip else "child"
        print(f"from: {name_from} to: {name_to}")
        msg = Message(msg=args.text)
        mhash = msg_store.write_message(msg)
        embed = embed_store.get_embedding(
            msg_store, name_from, mhash, no_index=precise, no_cache=no_cache)
        divide = "=" * 42
        print(f"query: {msg.get_text()}")
        for out in embed_store.get_closest(
                name_to, embed, 20, precise=precise, no_cache=no_cache):
            if args.count:
                continue
            print(divide)
            print(msg_store.read_message(out).get_text())
    elif args.count:
        count = msg_store.get_message_count()
        print(f"{count} messages available")
    else:
        if no_cache:
            print("nothing to do (remove --no-cache)")
            return
        embed_store.ensure_all(msg_store, no_index=precise)


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
        help=(
            "whether index lookup should be precise "
            "(this will prevent creation of an index)"))
    parser.add_argument(
        "--no-cache",
        default=False,
        action="store_true",
        help=(
            "does not cache embeddings "
            "(this will prevent cache from building, too)"))
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
    parser.add_argument(
        "--parent",
        default=False,
        action="store_true",
        help="search for a parent instead of a child")
    return parser.parse_args()


if __name__ == "__main__":
    run()
