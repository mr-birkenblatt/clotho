# pylint: disable=import-error,relative-beyond-top-level,unused-import
# pylint: disable=useless-import-alias,multiple-statements,no-name-in-module
# pylint: disable=unused-argument,invalid-name
from typing import Optional

from ....const import API_PATH as API_PATH

class SavableMixin:
    def save(self, *, category: Optional[str] = ...) -> None: ...
    def unsave(self) -> None: ...
