# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


from argparse import Action

from _typeshed import Incomplete


class env(Action):

    def __init__(
        self, dest, default: Incomplete | None = ..., required: bool = ...,
        **kwargs) -> None: ...

    def __call__(
        self, parser, namespace, values,
        option_string: Incomplete | None = ...) -> None: ...


class check_env(Action):
    def __init__(self, dest, default: bool = ..., **kwargs) -> None: ...

    def __call__(
        self, parser, namespace, values,
        option_string: Incomplete | None = ...) -> None: ...
