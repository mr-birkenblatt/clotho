# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


from _typeshed import Incomplete

from .optimizer import Optimizer as Optimizer


class LBFGS(Optimizer):

    def __init__(
        self, params, lr: int = ..., max_iter: int = ...,
        max_eval: Incomplete | None = ..., tolerance_grad: float = ...,
        tolerance_change: float = ..., history_size: int = ...,
        line_search_fn: Incomplete | None = ...) -> None: ...

    def step(self, closure): ...
