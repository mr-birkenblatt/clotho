# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


from _typeshed import Incomplete


class _RequiredParameter:
    ...


required: Incomplete


class Optimizer:
    defaults: Incomplete
    state: Incomplete
    param_groups: Incomplete
    def __init__(self, params, defaults) -> None: ...
    def state_dict(self): ...
    def load_state_dict(self, state_dict): ...
    def zero_grad(self, set_to_none: bool = ...): ...
    def step(self, closure) -> None: ...
    def add_param_group(self, param_group) -> None: ...
