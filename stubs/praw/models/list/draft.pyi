# pylint: disable=import-error,relative-beyond-top-level,unused-import
# pylint: disable=useless-import-alias,multiple-statements,no-name-in-module
# pylint: disable=unused-argument,invalid-name
# pylint: disable=too-few-public-methods
from .base import BaseList as BaseList

class DraftList(BaseList):
    CHILD_ATTRIBUTE: str
