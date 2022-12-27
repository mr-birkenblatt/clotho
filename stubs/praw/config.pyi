# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level
from typing import Optional

from _typeshed import Incomplete

from .exceptions import ClientException as ClientException


class _NotSet:
    def __bool__(self) -> bool: ...
    __nonzero__: Incomplete


class Config:
    CONFIG: Incomplete
    CONFIG_NOT_SET: Incomplete
    LOCK: Incomplete
    INTERPOLATION_LEVEL: Incomplete

    @property
    def short_url(self) -> str: ...

    custom: Incomplete
    client_id: Incomplete
    reddit_url: Incomplete
    password: Incomplete

    def __init__(
        self, site_name: str,
        config_interpolation: Optional[str] = ...,
        **settings: str) -> None: ...
