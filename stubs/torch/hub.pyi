# pylint: disable=multiple-statements,unused-argument, invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias, unused-import
# pylint: disable=redefined-builtin,super-init-not-called, arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors, import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member, keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name, c-extension-no-member
# pylint: disable=protected-access,no-name-in-module, undefined-variable


from typing import Any, Callable, Dict, Optional, Union

from _typeshed import Incomplete


class tqdm:
    total: Incomplete
    disable: Incomplete
    n: int

    def __init__(
        self, total: Incomplete | None = ..., disable: bool = ...,
        unit: Incomplete | None = ..., unit_scale: Incomplete | None = ...,
        unit_divisor: Incomplete | None = ...) -> None: ...

    def update(self, n) -> None: ...
    def close(self) -> None: ...
    def __enter__(self): ...
    def __exit__(self, exc_type, exc_val, exc_tb) -> None: ...


def get_dir(): ...


def set_dir(d) -> None: ...


def list(
    github, force_reload: bool = ..., skip_validation: bool = ...,
    trust_repo: Incomplete | None = ...): ...


def help(
    github, model, force_reload: bool = ..., skip_validation: bool = ...,
    trust_repo: Incomplete | None = ...): ...


def load(
    repo_or_dir, model, *args, source: str = ...,
    trust_repo: Incomplete | None = ..., force_reload: bool = ...,
    verbose: bool = ..., skip_validation: bool = ..., **kwargs): ...


def download_url_to_file(
    url, dst, hash_prefix: Incomplete | None = ...,
    progress: bool = ...) -> None: ...


def load_state_dict_from_url(
    url: str, model_dir: Optional[str] = ...,
    map_location: Optional[Union[Callable[[str], str], Dict[str,
                            str]]] = ..., progress: bool = ...,
    check_hash: bool = ..., file_name: Optional[str] = ...) -> Dict[
        str, Any]: ...
