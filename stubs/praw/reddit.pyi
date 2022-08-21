# pylint: disable=useless-import-alias,unused-import,multiple-statements
# pylint: disable=import-error,relative-beyond-top-level,no-name-in-module
# pylint: disable=unused-argument,redefined-builtin,invalid-name

from typing import Any, Dict, Generator, IO, Iterable, Optional, Type, Union

import praw
from _typeshed import Incomplete
from prawcore import Requestor  # type: ignore

from . import models as models
from .config import Config as Config
from .const import API_PATH as API_PATH
from .const import USER_AGENT_FORMAT as USER_AGENT_FORMAT
from .exceptions import ClientException as ClientException
from .exceptions import (
    MissingRequiredAttributeException as MissingRequiredAttributeException,
)
from .exceptions import RedditAPIException as RedditAPIException
from .objector import Objector as Objector
from .util.token_manager import BaseTokenManager as BaseTokenManager


UPDATE_CHECKER_MISSING: bool
Comment: Incomplete
Redditor: Incomplete
Submission: Incomplete
Subreddit: Incomplete
logger: Incomplete


class Reddit:
    update_checked: bool

    @property
    def read_only(self) -> bool: ...

    @read_only.setter
    def read_only(self, value: bool) -> None: ...

    @property
    def validate_on_submit(self) -> bool: ...

    @validate_on_submit.setter
    def validate_on_submit(self, val: bool) -> None: ...

    def __enter__(self) -> None: ...

    def __exit__(self, *args: Any) -> bool: ...

    config: Incomplete
    auth: Incomplete
    drafts: Incomplete
    front: Incomplete
    inbox: Incomplete
    live: Incomplete
    multireddit: Incomplete
    notes: Incomplete
    redditors: Incomplete
    subreddit: Incomplete
    subreddits: Incomplete
    user: Incomplete

    def __init__(
        self, site_name: Optional[str] = ..., *,
        config_interpolation: Optional[str] = ...,
        requestor_class: Optional[Type[Requestor]] = ...,
        requestor_kwargs: Optional[Dict[str, Any]] = ...,
        token_manager: Optional[BaseTokenManager] = ...,
        **config_settings: Optional[Union[str, bool]]) -> None: ...

    def comment(
        self, id: Optional[str] = ..., *,
        url: Optional[str] = ...) -> None: ...

    def domain(self, domain: str) -> None: ...

    def get(
        self, path: str, *,
        params: Optional[Union[str, Dict[str, Union[str, int]]]] = ...,
        ) -> None: ...

    def info(
        self, *, fullnames: Optional[Iterable[str]] = ...,
        subreddits:
            Optional[Iterable[Union['praw.models.Subreddit', str]]] = ...,
        url: Optional[str] = ...) -> Generator[
            Union[
                'praw.models.Subreddit',
                'praw.models.Comment',
                'praw.models.Submission'], None, None]: ...

    def delete(
        self, path: str, *, data: Optional[
            Union[Dict[str, Union[str, Any]], bytes, IO, str]] = ...,
        json: Optional[Dict[Any, Any]] = ...,
        params: Optional[Union[str, Dict[str, str]]] = ...) -> Any: ...

    def patch(
        self, path: str, *, data: Optional[
            Union[Dict[str, Union[str, Any]], bytes, IO, str]] = ...,
        json: Optional[Dict[Any, Any]] = ...) -> Any: ...

    def post(
        self, path: str, *, data: Optional[
            Union[Dict[str, Union[str, Any]], bytes, IO, str]] = ...,
        files: Optional[Dict[str, IO]] = ...,
        json: Optional[Dict[Any, Any]] = ...,
        params: Optional[Union[str, Dict[str, str]]] = ...) -> Any: ...

    def put(
        self, path: str, *, data: Optional[
            Union[Dict[str, Union[str, Any]], bytes, IO, str]] = ...,
        json: Optional[Dict[Any, Any]] = ...) -> None: ...

    def random_subreddit(
        self, *, nsfw: bool = ...) -> praw.models.Subreddit: ...

    def redditor(
        self, name: Optional[str] = ..., *,
        fullname: Optional[str] = ...) -> praw.models.Redditor: ...

    def request(
        self, *, data: Optional[
            Union[Dict[str, Union[str, Any]], bytes, IO, str]] = ...,
        files: Optional[Dict[str, IO]] = ...,
        json: Optional[Dict[Any, Any]] = ..., method: str,
        params: Optional[Union[str, Dict[str, Union[str, int]]]] = ...,
        path: str) -> Any: ...

    def submission(
        self,
        id: Optional[str] = ..., *,
        url: Optional[str] = ...) -> praw.models.Submission: ...

    def username_available(self, name: str) -> bool: ...
