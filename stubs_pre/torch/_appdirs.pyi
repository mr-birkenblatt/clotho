# pylint: disable=multiple-statements,unused-argument, invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias, unused-import
# pylint: disable=redefined-builtin,super-init-not-called, arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors, import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member, keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name, c-extension-no-member
# pylint: disable=protected-access,no-name-in-module, undefined-variable


from ctypes import windll as windll

from _typeshed import Incomplete


__version_info__: Incomplete
unicode = str
system: Incomplete


def user_data_dir(
    appname: Incomplete | None = ..., appauthor: Incomplete | None = ...,
    version: Incomplete | None = ..., roaming: bool = ...): ...


def site_data_dir(
    appname: Incomplete | None = ..., appauthor: Incomplete | None = ...,
    version: Incomplete | None = ..., multipath: bool = ...): ...


def user_config_dir(
    appname: Incomplete | None = ..., appauthor: Incomplete | None = ...,
    version: Incomplete | None = ..., roaming: bool = ...): ...


def site_config_dir(
    appname: Incomplete | None = ..., appauthor: Incomplete | None = ...,
    version: Incomplete | None = ..., multipath: bool = ...): ...


def user_cache_dir(
    appname: Incomplete | None = ..., appauthor: Incomplete | None = ...,
    version: Incomplete | None = ..., opinion: bool = ...): ...


def user_state_dir(
    appname: Incomplete | None = ..., appauthor: Incomplete | None = ...,
    version: Incomplete | None = ..., roaming: bool = ...): ...


def user_log_dir(
    appname: Incomplete | None = ..., appauthor: Incomplete | None = ...,
    version: Incomplete | None = ..., opinion: bool = ...): ...


class AppDirs:
    appname: Incomplete
    appauthor: Incomplete
    version: Incomplete
    roaming: Incomplete
    multipath: Incomplete

    def __init__(
        self, appname: Incomplete | None = ...,
        appauthor: Incomplete | None = ..., version: Incomplete | None = ...,
        roaming: bool = ..., multipath: bool = ...) -> None: ...

    @property
    def user_data_dir(self): ...
    @property
    def site_data_dir(self): ...
    @property
    def user_config_dir(self): ...
    @property
    def site_config_dir(self): ...
    @property
    def user_cache_dir(self): ...
    @property
    def user_state_dir(self): ...
    @property
    def user_log_dir(self): ...
