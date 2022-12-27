# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


from _typeshed import Incomplete

from .base_scheduler import BaseScheduler as BaseScheduler


class LambdaSL(BaseScheduler):
    sparsifier: Incomplete
    sl_lambdas: Incomplete

    def __init__(
        self, sparsifier, sl_lambda, last_epoch: int = ...,
        verbose: bool = ...) -> None: ...

    def get_sl(self): ...
