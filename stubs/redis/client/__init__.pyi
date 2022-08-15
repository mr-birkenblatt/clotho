# pylint: disable=too-few-public-methods,unused-argument


from typing import List, Optional, Union

from redis import StrictRedis

class Script:
    def __init__(
            self,
            registered_client: Optional[StrictRedis],
            script: Union[bytes, str]) -> None:
        ...

    def __call__(
            self,
            keys: List[str] = ...,
            args: List[Union[bytes, str, int]] = ...,
            client: Optional[StrictRedis] = ...) -> bytes:
        ...
