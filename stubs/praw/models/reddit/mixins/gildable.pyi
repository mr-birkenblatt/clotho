# pylint: disable=import-error,relative-beyond-top-level,unused-import
# pylint: disable=useless-import-alias,multiple-statements,no-name-in-module
# pylint: disable=unused-argument,invalid-name
from ....const import API_PATH as API_PATH


class GildableMixin:
    def award(
        self, *, gild_type: str = ..., is_anonymous: bool = ...,
        message: str = ...) -> dict: ...

    def gild(self) -> None: ...
