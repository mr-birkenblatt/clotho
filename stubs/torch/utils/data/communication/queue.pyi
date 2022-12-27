# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


from _typeshed import Incomplete


class LocalQueue:
    ops: int
    stored: int
    uid: int
    empty: int
    items: Incomplete
    name: Incomplete
    def __init__(self, name: str = ...) -> None: ...
    def put(self, item, block: bool = ...) -> None: ...
    def get(self, block: bool = ..., timeout: int = ...): ...


class ThreadingQueue:
    lock: Incomplete
    items: Incomplete
    name: Incomplete
    def __init__(self, name: str = ...) -> None: ...
    def put(self, item, block: bool = ...) -> None: ...
    def get(self, block: bool = ..., timeout: int = ...): ...
