from typing import Optional

from ....const import API_PATH as API_PATH

class SavableMixin:
    def save(self, *, category: Optional[str] = ...): ...
    def unsave(self) -> None: ...
