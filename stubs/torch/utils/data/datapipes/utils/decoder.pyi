# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access,no-name-in-module,undefined-variable


from _typeshed import Incomplete


def basichandlers(extension, data): ...


def handle_extension(extensions, f): ...


class ImageHandler:
    imagespec: Incomplete
    def __init__(self, imagespec) -> None: ...
    def __call__(self, extension, data): ...


def imagehandler(imagespec): ...


def videohandler(extension, data): ...


def audiohandler(extension, data): ...


class MatHandler:
    sio: Incomplete
    loadmat_kwargs: Incomplete
    def __init__(self, **loadmat_kwargs) -> None: ...
    def __call__(self, extension, data): ...


def mathandler(**loadmat_kwargs): ...


def extension_extract_fn(pathname): ...


class Decoder:
    handlers: Incomplete
    key_fn: Incomplete
    def __init__(self, *handler, key_fn=...) -> None: ...
    def add_handler(self, *handler) -> None: ...
    def decode1(self, key, data): ...
    def decode(self, data): ...
    def __call__(self, data): ...
