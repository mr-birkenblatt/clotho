# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


from _typeshed import Incomplete
from torch.ao.sparsity import BaseSparsifier as BaseSparsifier


class BaseScheduler:
    sparsifier: Incomplete
    base_sl: Incomplete
    last_epoch: Incomplete
    verbose: Incomplete

    def __init__(
        self, sparsifier, last_epoch: int = ..., verbose: bool = ...): ...

    def state_dict(self): ...
    def load_state_dict(self, state_dict) -> None: ...
    def get_last_sl(self): ...
    def get_sl(self) -> None: ...

    def print_sl(
        self, is_verbose, group, sl,
        epoch: Incomplete | None = ...) -> None: ...

    o: Incomplete
    def step(self, epoch: Incomplete | None = ...): ...
