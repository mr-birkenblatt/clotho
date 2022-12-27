# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access,no-name-in-module,undefined-variable
from typing import Any

from _typeshed import Incomplete


def get_cudnn_mode(mode: Any) -> Any: ...


class Unserializable:
    inner: Incomplete
    def __init__(self, inner: Any) -> None: ...
    def get(self) -> Any: ...


def init_dropout_state(
    dropout: Any, train: Any,
    dropout_seed: Any, dropout_state: Any) -> Any: ...
