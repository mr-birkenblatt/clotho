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
            redirect_uri="http://localhost:8080",
            user_agent=f"testscript by u/{creds['user']}",
        )
        print(
            reddit.auth.url(
                scopes=["identity"], state="...", duration="permanent"))
