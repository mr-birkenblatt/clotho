# pylint: disable=import-error,relative-beyond-top-level,unused-import
# pylint: disable=useless-import-alias,multiple-statements,no-name-in-module
# pylint: disable=unused-argument,invalid-name
from typing import Any, Dict, Iterator, List, Optional, Union

import praw
from _typeshed import Incomplete

from ...const import API_PATH as API_PATH
from ...exceptions import ClientException as ClientException
from ...util import cachedproperty as cachedproperty
from .base import RedditBase as RedditBase


class Rule(RedditBase):
    STR_FIELD: str
    def mod(self) -> praw.models.reddit.rules.RuleModeration: ...
    short_name: Incomplete
    subreddit: Incomplete

    def __init__(
        self, reddit: praw.Reddit,
        subreddit: Optional['praw.models.Subreddit'] = ...,
        short_name: Optional[str] = ...,
        _data: Optional[Dict[str, str]] = ...) -> None: ...

    def __getattribute__(self, attribute: str) -> Any: ...


class SubredditRules:
    def mod(self) -> 'SubredditRulesModeration': ...
    def __call__(self) -> List['praw.models.Rule']: ...

    def __getitem__(
        self, short_name: Union[str, int, slice]) -> praw.models.Rule: ...

    subreddit: Incomplete
    def __init__(self, subreddit: praw.models.Subreddit) -> None: ...
    def __iter__(self) -> Iterator['praw.models.Rule']: ...


class RuleModeration:
    rule: Incomplete
    def __init__(self, rule: praw.models.Rule) -> None: ...
    def delete(self) -> None: ...

    def update(
        self, *, description: Optional[str] = ..., kind: Optional[str] = ...,
        short_name: Optional[str] = ..., violation_reason: Optional[str] = ...,
        ) -> praw.models.Rule: ...


class SubredditRulesModeration:
    subreddit_rules: Incomplete
    def __init__(self, subreddit_rules: SubredditRules) -> None: ...

    def add(
        self, *, description: str = ..., kind: str, short_name: str,
        violation_reason: Optional[str] = ...) -> praw.models.Rule: ...

    def reorder(
        self, rule_list: List['praw.models.Rule'],
        ) -> List['praw.models.Rule']: ...
