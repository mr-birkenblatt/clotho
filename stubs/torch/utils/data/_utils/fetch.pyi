# pylint: disable=multiple-statements,unused-argument,invalid-name
# pylint: disable=too-few-public-methods,useless-import-alias,unused-import
# pylint: disable=redefined-builtin,super-init-not-called,arguments-renamed
# pylint: disable=abstract-method,too-many-ancestors,import-error
# pylint: disable=relative-beyond-top-level,redefined-outer-name
# pylint: disable=arguments-differ,no-member,keyword-arg-before-vararg
# pylint: disable=signature-differs,blacklisted-name,c-extension-no-member
# pylint: disable=protected-access,no-name-in-module,undefined-variable


from _typeshed import Incomplete


class _BaseDatasetFetcher:
    dataset: Incomplete
    auto_collation: Incomplete
    collate_fn: Incomplete
    drop_last: Incomplete

    def __init__(
        self, dataset, auto_collation, collate_fn, drop_last) -> None: ...

    def fetch(self, possibly_batched_index) -> None: ...


class _IterableDatasetFetcher(_BaseDatasetFetcher):
    dataset_iter: Incomplete
    ended: bool

    def __init__(
        self, dataset, auto_collation, collate_fn, drop_last) -> None: ...

    def fetch(self, possibly_batched_index): ...


class _MapDatasetFetcher(_BaseDatasetFetcher):

    def __init__(
        self, dataset, auto_collation, collate_fn, drop_last) -> None: ...

    def fetch(self, possibly_batched_index): ...
