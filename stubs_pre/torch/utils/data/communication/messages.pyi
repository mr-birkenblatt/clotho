# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access,no-name-in-module,undefined-variable


from _typeshed import Incomplete


class DataLoaderQueueMessage:
    ...


class Request(DataLoaderQueueMessage):
    ...


class Response(DataLoaderQueueMessage):
    ...


class ResetIteratorRequest(Request):
    ...


class ResetIteratorResponse(Response):
    ...


class TerminateRequest(Request):
    ...


class TerminateResponse(Response):
    ...


class LenRequest(Request):
    ...


class LenResponse(Response):
    len: Incomplete
    def __init__(self, len) -> None: ...


class GetItemRequest(Request):
    key: Incomplete
    def __init__(self, key) -> None: ...


class GetItemResponse(Response):
    key: Incomplete
    value: Incomplete
    def __init__(self, key, value) -> None: ...


class GetNextRequest(Request):
    ...


class GetNextResponse(Response):
    value: Incomplete
    def __init__(self, value) -> None: ...


class StopIterationResponse(Response):
    ...


class InvalidStateResponse(Response):
    ...
