# pylint: disable=import-error,relative-beyond-top-level,unused-import
# pylint: disable=useless-import-alias,multiple-statements,unused-argument
from typing import Dict, List, Optional, Set, Union

from ..exceptions import InvalidImplicitAuth as InvalidImplicitAuth
from ..exceptions import (
    MissingRequiredAttributeException as MissingRequiredAttributeException,
)
from .base import PRAWBase as PRAWBase

class Auth(PRAWBase):
    @property
    def limits(self) -> Dict[str, Optional[Union[str, int]]]: ...
    def authorize(self, code: str) -> Optional[str]: ...
    def implicit(
        self, *, access_token: str, expires_in: int, scope: str) -> None: ...

    def scopes(self) -> Set[str]: ...

    def url(
        self, *, duration: str = ..., implicit: bool = ...,
        scopes: List[str], state: str) -> str: ...
