from _typeshed import Incomplete

class InlineMedia:
    TYPE: Incomplete
    path: Incomplete
    caption: Incomplete
    media_id: Incomplete
    def __init__(self, *, caption: str = ..., path: str) -> None: ...
    def __eq__(self, other: InlineMedia): ...

class InlineGif(InlineMedia):
    TYPE: str

class InlineVideo(InlineMedia):
    TYPE: str

class InlineImage(InlineMedia):
    TYPE: str
