from _typeshed import Incomplete


DEFAULT_EXTRA_FILE_SIZE_LIMIT: Incomplete


def get_storage_info(storage): ...


def hierarchical_pickle(data): ...


def get_model_info(
    path_or_file,
    title: Incomplete | None = ..., extra_file_size_limit=...): ...


def get_inline_skeleton(): ...


def burn_in_info(skeleton, info): ...


def get_info_and_burn_skeleton(path_or_bytesio, **kwargs): ...


def main(argv, *, stdout: Incomplete | None = ...) -> None: ...
