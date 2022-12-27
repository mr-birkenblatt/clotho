# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access,no-name-in-module,undefined-variable


from typing import Optional, Union

import torch


class _remote_device:
    def __init__(self, remote_device: Union[str, torch.device]) -> None: ...
    def worker_name(self) -> Optional[str]: ...
    def rank(self) -> Optional[int]: ...
    def device(self) -> torch.device: ...
    def __eq__(self, other): ...
    def __hash__(self): ...
