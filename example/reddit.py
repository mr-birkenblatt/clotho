import collections
import os
from typing import (
    Deque,
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    Tuple,
    TypedDict,
    Union,
)

from praw import Reddit
from praw.models.comment_forest import CommentForest
from praw.models.reddit.comment import Comment
from praw.models.reddit.more import MoreComments
from praw.models.reddit.submission import Submission
from praw.models.reddit.subreddit import Subreddit

from misc.io import open_read
from misc.util import json_load, json_pretty


MaybeComment = Union[Comment, MoreComments]
CommentsOrForest = Union[CommentForest, List[MaybeComment]]


CRED_FILE = os.path.join(os.path.dirname(__file__), "creds.json")

MessageAction = TypedDict('MessageAction', {
    "text": str,
})
LinkAction = TypedDict('LinkAction', {
    "parent_ref": str,
    "user": str,
    "user_ref": str,
    "created_utc": float,
    "votes": Dict[str, int],
})
Action = TypedDict('Action', {
    "kind": Literal["message", "link"],
    "ref_id": str,
    "message": Optional[MessageAction],
    "link": Optional[LinkAction],
}, total=False)


def create_message_action(ref_id: str, text: str) -> Action:
    return {
        "kind": "message",
        "ref_id": ref_id,
        "message": {
            "text": text,
        },
    }


def create_link_action(
        ref_id: str,
        parent_ref: str,
        user_ref: str,
        user: str,
        created_utc: float,
        votes: Dict[str, int]) -> Action:
    return {
        "kind": "link",
        "ref_id": ref_id,
        "link": {
            "parent_ref": parent_ref,
            "user": user,
            "user_ref": user_ref,
            "created_utc": created_utc,
            "votes": votes,
        },
    }


class RedditAccess:
    def __init__(self) -> None:
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
        self._users: Dict[str, Tuple[str, str]] = {}

    def get_posts(self, subreddit: str) -> Iterable[Submission]:
        yield from self._reddit.subreddit(subreddit).hot()

    def get_user(self, value: Union[Submission, Comment]) -> Tuple[str, str]:
        if not hasattr(value, "author_fullname"):
            return ("NOUSER", "NOUSER")
        ref = value.author_fullname
        res = self._users.get(ref, None)
        if res is None:
            user = value.author
            user_ref = "NOUSER" if user is None else user.fullname
            user_name = "NOUSER" if user is None else f"u/{user.name}"
            res = (user_ref, user_name)
            self._users[ref] = res
        return res

    def create_link_action(
            self, parent_id: str, value: Union[Submission, Comment]) -> Action:
        user_ref, user_name = self.get_user(value)
        ups = max(value.ups, 0) - min(value.downs, 0)
        downs = max(value.downs, 0) - min(value.ups, 0)
        votes = {
            "up": ups,
            "down": downs,
        }
        if value.total_awards_received > 0:
            print(f"awards ({value.total_awards_received})")
            votes.update({
                award["name"]: award["count"]
                for award in value.all_awardings
            })
        return create_link_action(
            value.fullname,
            parent_id,
            user_ref,
            user_name,
            value.created_utc,
            votes)

    @staticmethod
    def create_message_action(
            value: Union[Subreddit, Submission, Comment]) -> Action:
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

    def get_comments(self, sid: Union[str, Submission]) -> Iterable[Action]:
        if isinstance(sid, str):
            doc: Submission = self._reddit.submission(id=sid)
        else:
            doc = sid

        sub = doc.subreddit
        yield self.create_message_action(sub)
        yield self.create_message_action(doc)
        yield self.create_link_action(sub.fullname, doc)

        queue: Deque[Tuple[str, CommentsOrForest]] = collections.deque()
        queue.append((doc.fullname, doc.comments))

        def process(
                parent_id: str, curs: CommentsOrForest) -> Iterable[Action]:
            print(f"batch ({len(curs)})")
            for comment in curs:
                if isinstance(comment, MoreComments):
                    queue.append((parent_id, comment.comments()))
                    continue
                if comment.replies:
                    queue.append((comment.fullname, comment.replies))
                yield self.create_message_action(comment)
                yield self.create_link_action(parent_id, comment)

        while queue:
            yield from process(*queue.popleft())
