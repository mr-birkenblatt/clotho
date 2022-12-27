# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


from typing import List


def get_all(store, rank: int, prefix: str, size: int): ...


def synchronize(
    store, data: bytes, rank: int, world_size: int, key_prefix: str,
        barrier_timeout: float = ...) -> List[bytes]: ...


def barrier(
    store, rank: int, world_size: int, key_prefix: str,
        barrier_timeout: float = ...) -> None: ...
