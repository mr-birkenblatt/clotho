import collections
import logging
import os
import time
from typing import Iterable

import praw
from praw import Reddit
from praw.models.comment_forest import CommentForest
from praw.models.reddit.comment import Comment
from praw.models.reddit.more import MoreComments
from praw.models.reddit.submission import Submission
from praw.models.reddit.subreddit import Subreddit

from example.action import Action, create_link_action, create_message_action
from misc.io import open_read
from misc.util import json_load, json_pretty


MaybeComment = Comment | MoreComments
CommentsOrForest = CommentForest | list[MaybeComment]


CRED_FILE = os.path.join(os.path.dirname(__file__), "creds.json")


class RedditAccess:
    def __init__(self, do_log: bool) -> None:
        if do_log:
            handler = logging.StreamHandler()
            handler.setLevel(logging.DEBUG)
            for logger_name in ("praw", "prawcore"):
                logger = logging.getLogger(logger_name)
                logger.setLevel(logging.DEBUG)
                logger.addHandler(handler)
        with open_read(CRED_FILE, text=True) as fin:
            creds = json_load(fin)
        if "ERROR" in creds.values():
            if creds["client_secret"] != "ERROR":
                creds["client_secret"] = "???"
            raise ValueError(f"invalid creds {json_pretty(creds)}")
        reddit = Reddit(
            client_id=creds["client_id"],
            client_secret=creds["client_secret"],
            user_agent=f"testscript by u/{creds['user']}")
        self._reddit = reddit
        self._users: dict[str, tuple[str, str]] = {}

    def get_posts(
            self, subreddit: str, *, is_top: bool) -> Iterable[Submission]:
        if is_top:
            yield from self._reddit.subreddit(subreddit).top(limit=25)
        else:
            yield from self._reddit.subreddit(subreddit).hot()

    def get_user(self, value: Submission | Comment) -> tuple[str, str]:
        if not hasattr(value, "author_fullname"):
            return ("NOUSER", "u/NOUSER")
        ref = value.author_fullname
        res = self._users.get(ref, None)
        if res is None:
            user = value.author
            user_ref = getattr(user, "fullname", "NOUSER")
            user_name = f"u/{getattr(user, 'name', 'NOUSER')}"
            res = (user_ref, user_name)
            self._users[ref] = res
        return res

    def create_link_action(
            self,
            parent_id: str,
            depth: int,
            value: Submission | Comment) -> Action:
        user_ref = getattr(value, "author_fullname", None)
        if user_ref is not None:
            user = value.author
            user_name = f"u/{getattr(user, 'name', 'NOUSER')}"
        else:
            user_name = None
        ups = max(value.ups, 0) - min(value.downs, 0)
        downs = max(value.downs, 0) - min(value.ups, 0)
        votes = {
            "up": ups,
            "down": downs,
        }
        if value.total_awards_received > 0:
            # print(f"awards ({value.total_awards_received})")
            votes.update({
                award["name"]: award["count"]
                for award in value.all_awardings
            })
        return create_link_action(
            value.fullname,
            parent_id,
            depth,
            user_ref,
            user_name,
            value.created_utc,
            votes)

    @staticmethod
    def create_message_action(
            value: Subreddit | Submission | Comment) -> Action:
        if isinstance(value, Subreddit):
            sub: Subreddit = value
            return create_message_action(sub.fullname, f"r/{sub.display_name}")
        if isinstance(value, Submission):
            doc: Submission = value
            if doc.is_self:
                text = f"{doc.title}\n{doc.selftext}"
            else:
                text = f"[{doc.title}]({doc.url})"
            return create_message_action(doc.fullname, text)
        if isinstance(value, Comment):
            comment: Comment = value
            return create_message_action(comment.fullname, comment.body)
        raise TypeError(f"invalid type of {value}: {type(value)}")

    def get_comments(self, sid: str | Submission) -> Iterable[Action]:
        if isinstance(sid, str):
            doc: Submission = self._reddit.submission(id=sid)
        else:
            doc = sid

        sub = doc.subreddit
        yield self.create_message_action(sub)
        yield self.create_message_action(doc)
        yield self.create_link_action(sub.fullname, -1, doc)

        timing_start = time.monotonic()
        already: set[str] = set()

        queue: collections.deque[CommentsOrForest] = collections.deque()
        queue.append(doc.comments)

        def process(curs: CommentsOrForest) -> Iterable[Action]:
            mores: list[MoreComments] = []
            if isinstance(curs, CommentForest):
                try:
                    curs.replace_more(limit=None)
                except praw.exceptions.DuplicateReplaceException:
                    print(f"double replace: {curs}")
                curs = curs.list()
                if curs:
                    print(
                        f"batch ({len(curs)}) "
                        f"{time.monotonic() - timing_start:.2f}s")
            for comment in curs:
                if isinstance(comment, MoreComments):
                    mores.append(comment)
                    continue
                cur_ref = comment.fullname
                if cur_ref in already:
                    print(f"redundant comment: {cur_ref}")
                    continue
                # NOTE: not worth it
                # try:
                #     comment.refresh()
                # except praw.exceptions.ClientException:
                #     print(f"comment doesn't exist: {comment.fullname}")
                # if comment.replies:
                #     queue.append(comment.replies)
                yield self.create_message_action(comment)
                yield self.create_link_action(
                    comment.parent_id, comment.depth, comment)
                already.add(cur_ref)
            for more in mores:
                queue.append(more.comments())

        while queue:
            yield from process(queue.popleft())

        print(f"done ({len(already)}) {time.monotonic() - timing_start:.2f}s")
