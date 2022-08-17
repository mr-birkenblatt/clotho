# pylint: disable=too-few-public-methods

class FullnameMixin:
    @property
    def fullname(self) -> str:
        ...
