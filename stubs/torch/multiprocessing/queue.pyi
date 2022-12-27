# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access,no-name-in-module,undefined-variable


import multiprocessing.queues

from _typeshed import Incomplete


class ConnectionWrapper:
    conn: Incomplete
    def __init__(self, conn) -> None: ...
    def send(self, obj) -> None: ...
    def recv(self): ...
    def __getattr__(self, name): ...


class Queue(multiprocessing.queues.Queue):
    def __init__(self, *args, **kwargs) -> None: ...


class SimpleQueue(multiprocessing.queues.SimpleQueue):
    ...
