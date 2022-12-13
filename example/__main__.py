import argparse
import os
import time

import pandas as pd

from effects.effects import get_old_threshold, set_old_threshold
from example.loader import process_action_file
from example.reddit import RedditAccess
from misc.io import open_append
from misc.util import json_compact
from system.links.store import get_link_store
from system.msgs.store import get_message_store
from system.namespace.store import get_namespace
from system.users.store import get_user_store


REDDIT_ACTION_FILE = os.path.join(os.path.dirname(__file__), "reddit.jsonl")
# ROOTS = ["politics", "news", "worldnews", "conservative"]
ROOTS = ["askscience", "askreddit", "explainlikeimfive", "todayilearned"]


def process_reddit(reddit: RedditAccess, fname: str, subs: list[str]) -> None:
    dups: set[str] = set()
    with open_append(fname, text=True) as fout:
        for sub in subs:
            for doc in reddit.get_posts(sub):
                print(
                    f"processing {doc.subreddit_name_prefixed} "
                    f"\"{doc.title}\" (est. comments {doc.num_comments})")
                if doc.is_meta or doc.is_created_from_ads_ui or doc.pinned:
                    print(
                        f"skipping is_meta={doc.is_meta} "
                        f"is_created_from_ads_ui={doc.is_created_from_ads_ui} "
                        f"pinned={doc.pinned}")
                    continue
                for action in reddit.get_comments(doc):
                    a_str = json_compact(action).decode("utf-8")
                    a_str = a_str.replace("\n", "\\n")
                    if a_str in dups:
                        print(f"skip duplicate action {a_str}")
                        continue
                    print(a_str, file=fout)
                    # dups.add(a_str)  # NOTE: not worth it!


def process_load(ns_name: str) -> None:
    namespace = get_namespace(ns_name)
    message_store = get_message_store(namespace)
    link_store = get_link_store(namespace)
    user_store = get_user_store(namespace)
    now = pd.Timestamp("2022-08-22", tz="UTC")
    reference_time = time.monotonic()
    old_th = get_old_threshold()
    set_old_threshold(24 * 60 * 60)
    process_action_file(
        REDDIT_ACTION_FILE,
        message_store=message_store,
        link_store=link_store,
        user_store=user_store,
        now=now,
        reference_time=reference_time,
        roots=set(ROOTS))
    set_old_threshold(old_th)


def run() -> None:
    args = parse_args()
    if args.cmd == "reddit":
        process_reddit(RedditAccess(do_log=False), REDDIT_ACTION_FILE, ROOTS)
    elif args.cmd == "load":
        process_load(args.namespace)
    else:
        raise RuntimeError(f"invalid cmd: {args.cmd}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog=f"python -m {os.path.basename(os.path.dirname(__file__))}",
        description="Create or load example")
    parser.add_argument(
        "cmd",
        choices=["reddit", "load"],
        help="the command to execute")
    parser.add_argument("--namespace", default="default", help="the namespace")
    return parser.parse_args()


if __name__ == "__main__":
    run()
