# pylint: disable=multiple-statements,unused-argument, invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias, unused-import
# pylint: disable=redefined-builtin,super-init-not-called, arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors, import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member, keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name, c-extension-no-member
# pylint: disable=protected-access,no-name-in-module, undefined-variable


from _typeshed import Incomplete


class _LazyImport:
    def __init__(self, cls_name: str) -> None: ...
    def get_cls(self): ...
    def __call__(self, *args, **kwargs): ...
    def __instancecheck__(self, obj): ...


Version: Incomplete
InvalidVersion: Incomplete


class TorchVersion(str):
    ...
