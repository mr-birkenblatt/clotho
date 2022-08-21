import os

import praw

from misc.io import open_read
from misc.util import json_load, json_pretty


CRED_FILE = os.path.join(os.path.dirname(__file__), "creds.json")


class RedditAccess:
    def __init__(self) -> None:
        with open_read(CRED_FILE, text=True) as fin:
            creds = json_load(fin)
        if "ERROR" in creds.values():
            if creds["client_secret"] != "ERROR":
                creds["client_secret"] = "???"
            raise ValueError(f"invalid creds {json_pretty(creds)}")
        reddit = praw.Reddit(
            client_id=creds["client_id"],
            client_secret=creds["client_secret"],
            user_agent=f"testscript by u/{creds['user']}")
        print(reddit.read_only)
        for submission in reddit.subreddit("worldnews").hot(limit=10):
            print(type(submission))
            print(dir(submission))
            print(submission.title)
        self._reddit = reddit

    def get_posts(self) -> None:
        pass

    def get_comments(self) -> None:
        pass
