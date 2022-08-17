# pylint: disable=import-error,relative-beyond-top-level,unused-import
# pylint: disable=useless-import-alias,multiple-statements,no-name-in-module
# pylint: disable=unused-argument,invalid-name
from typing import Any, Optional

from ...util import cachedproperty as cachedproperty
from ..base import PRAWBase as PRAWBase

class PollOption(PRAWBase):
    ...


class PollData(PRAWBase):
    def user_selection(self) -> Optional[PollOption]: ...
    def __setattr__(self, attribute: str, value: Any) -> None: ...
    def option(self, option_id: str) -> PollOption: ...
