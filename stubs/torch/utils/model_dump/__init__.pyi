# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access


from _typeshed import Incomplete


DEFAULT_EXTRA_FILE_SIZE_LIMIT: Incomplete


def get_storage_info(storage): ...


def hierarchical_pickle(data): ...


def get_model_info(
    path_or_file, title: Incomplete | None = ...,
    extra_file_size_limit=...): ...


def get_inline_skeleton(): ...


def burn_in_info(skeleton, info): ...


def get_info_and_burn_skeleton(path_or_bytesio, **kwargs): ...


def main(argv, *, stdout: Incomplete | None = ...) -> None: ...
