import os
from typing import List, Set

from example.reddit import RedditAccess
from misc.io import open_append
from misc.util import json_compact


REDDIT_ACTION_FILE = os.path.join(os.path.dirname(__file__), "reddit.jsonl")


def process(reddit: RedditAccess, fname: str, subs: List[str]) -> None:
    dups: Set[str] = set()
    with open_append(fname, text=True) as fout:
        for sub in subs:
            for doc in reddit.get_posts(sub):
                print(
                    f"processing {doc.subreddit_name_prefixed} "
                    f"\"{doc.title}\" ({doc.num_comments})")
                for action in reddit.get_comments(doc):
                    a_str = json_compact(action).decode("utf-8")
                    a_str = a_str.replace("\\", "\\\\").replace("\n", "\\n")
                    if a_str in dups:
                        continue
                    print(a_str, file=fout)
                    dups.add(a_str)


if __name__ == "__main__":
    process(RedditAccess(), REDDIT_ACTION_FILE, ["worldnews", "news"])
