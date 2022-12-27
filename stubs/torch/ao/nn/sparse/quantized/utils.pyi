# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


from _typeshed import Incomplete


def is_valid_linear_block_sparse_pattern(row_block_size, col_block_size): ...


class LinearBlockSparsePattern:
    rlock: Incomplete
    row_block_size: int
    col_block_size: int
    prev_row_block_size: int
    prev_col_block_size: int

    def __init__(
        self, row_block_size: int = ...,
        col_block_size: int = ...) -> None: ...

    def __enter__(self) -> None: ...
    def __exit__(self, exc_type, exc_value, backtrace) -> None: ...
    @staticmethod
    def block_size(): ...
